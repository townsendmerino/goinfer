package decoder

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// DFlashDrafter is an imported z-lab DFlash block drafter (docs/spec/08): a small non-causal transformer trunk that
// reads the target's hidden states (the ForwardCapture seam) and proposes a whole block of tokens in one pass.
//
// The z-lab checkpoint ships only the trunk (decoder layers, fc, two norms), with no embedding, LM head, Markov head
// or confidence head: the block is embedded with the target's embed_tokens and the draft logits come out of the
// target's lm_head, so the caller supplies both ends. Drafters that do ship their own head and embedding use the
// lmHead and embed fields below.
//
// One round, given the target's captured hidden states for the committed context:
//
//	fused    = hiddenNorm(fc · concat(h[l] for l in targetLayerIDs))   per context position
//	blockIn  = targetEmbed([anchor, MASK, MASK, ...])                  [blockSize, hidden]
//	trunk    = layers(blockIn, attending over concat(fused, block))    bidirectional
//	logits   = targetLMHead(norm(trunk)[1:])                           blockSize-1 drafts
//
// The attention is cross-attention and non-causal: queries come only from the block, keys/values from the fused
// context concatenated with the block, and every block position sees every other. That is what makes the block one
// pass, and why the block width is not truncatable: each position's hidden depends on all of them, so a narrower block
// is off the trained distribution.
type DFlashDrafter struct {
	blockTrunk
	blockSize, maskTokenID int
	targetLayerIDs         []int

	// Own head and embedding, present only on drafters that ship them (poolside's Laguna speculators; z-lab's borrow the
	// target's, see DrafterHeadLogits). When lmHead is set the drafter emits ids in its own reduced vocabulary, which d2t
	// maps back to target ids: target_id = i + d2t[i] for draft index i. embed is the drafter's own token embedding over
	// the target vocab (it is fed target ids and produces the trunk's block input).
	lmHead     linalg.WeightMat // [draftVocab, hidden]; zero-valued when the drafter borrows the target's
	embed      linalg.WeightMat // [vocab, hidden]; zero-valued when it borrows the target's
	d2t        []int32          // draft index → target id OFFSET
	draftVocab int

	st *embed.SafetensorsFile // retained: the WeightMats alias its mmap; nil on a streamed drafter once loaded
	sf *stFile                // a streamed drafter's row readers' file (OpenDFlashDrafter); nil otherwise
}

// HasOwnHead reports whether this drafter emits ids from its own reduced vocabulary
// (and therefore must map them through d2t) rather than borrowing the target's LM
// head. Callers use it to pick between DraftTokenID and DrafterHeadLogits.
func (d *DFlashDrafter) HasOwnHead() bool { return d.draftVocab > 0 }

// DraftTokenID turns one already-normed trunk hidden row into a target token id using the drafter's own reduced-vocab
// head, mapping the argmax back through d2t. An unmapped argmax is a valid-looking id in the wrong space. Drafting is
// lossless (the target re-verifies every token), so it would not corrupt output, but it would silently destroy
// acceptance. Hence the mapping lives here, next to the argmax, not in the caller.
func (d *DFlashDrafter) DraftTokenID(be Backend, h []float32) int {
	d.mustHoldWeights()
	logits := make([]float32, d.draftVocab)
	matmul(be, &d.lmHead, h, logits, 1)
	i := argmax(logits)
	return i + int(d.d2t[i])
}

// EmbedDraft writes the drafter's OWN embedding row for a target token id. Drafters
// without their own embedding use the target's (Model.embedResident).
func (d *DFlashDrafter) EmbedDraft(id int, dst []float32) {
	d.mustHoldWeights()
	d.embed.Row(id, dst)
}

// EmbedBlock builds the trunk's block input for ids, using the drafter's own embedding when it has one and the
// target's otherwise. The fallback is not cosmetic: a drafter that ships embed_tokens was trained against those
// vectors, and the target's are a silent off-distribution input. The trunk still runs and output stays lossless, so
// the only symptom is acceptance landing below what the pairing is worth.
func (d *DFlashDrafter) EmbedBlock(m *Model, ids []int) [][]float32 {
	d.mustHoldWeights() // a released embedding is zero-rowed, and would read as "borrows the target's"
	if d.embed.Rows() == 0 {
		return m.DrafterEmbedBlock(ids)
	}
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		rows[i] = make([]float32, d.hidden)
		d.EmbedDraft(id, rows[i])
	}
	return rows
}

// DraftIDs turns the trunk's block output rows into TARGET token ids, using the
// drafter's own reduced-vocab head when present and the target's LM head otherwise.
// One call site for both drafter shapes, so a pairing cannot silently take the wrong
// output path.
func (d *DFlashDrafter) DraftIDs(m *Model, rows [][]float32) []int {
	d.mustHoldWeights()
	out := make([]int, len(rows))
	for i, h := range rows {
		if d.HasOwnHead() {
			out[i] = d.DraftTokenID(m.be, h)
		} else {
			out[i] = argmax(m.DrafterHeadLogits(h))
		}
	}
	return out
}

// blockTrunk is the non-causal block trunk shared by DFlash and DSpark, which compute it identically: hidden_norm(fc(ctx)),
// rotary, N layers, norm; attention is non-causal (is_causal=False) with K/V from concat(raw fused context, block); q
// takes cos[..., -q_len:, :] while k takes the full cos. What differs lives in the enclosing type: DFlash borrows the
// target's embedding and LM head and predicts from slot 1; DSpark ships its own, predicts from slot 0, and adds a
// rank-256 Markov chain plus a confidence head.
type blockTrunk struct {
	hidden, nHeads, nKV, headDim, inter int
	normEps                             float64
	invFreq                             []float64

	fc         linalg.WeightMat // [hidden, nTaps*hidden]
	hiddenNorm []float32        // RMSNorm on the fused context
	finalNorm  []float32        // RMSNorm at the end of the trunk
	layers     []dflashLayer

	layerCausal []bool // each layer's mask (DrafterGeometry.LayerCausal); nil: every layer full attention
	layerWindow []int

	released bool // releaseHostWeights ran: the matrices and norms above are zero

	// rows is each matrix's row reader, keyed by the field it stands in for, on a drafter opened to stream its matrices
	// (OpenDFlashDrafter): those fields hold nothing. Nil when the matrices are held.
	rows map[*linalg.WeightMat]DrafterRows
}

// ErrDrafterReleased is what host drafting on a drafter whose host weights were released returns (DFlashDrafter.ReleaseHostWeights),
// and what the methods with no error result panic with: its matrices are zero, and drafting on them would return garbage.
var ErrDrafterReleased = errors.New("decoder: the drafter's host weights were released after its resident attach; load it again to draft on the host")

// ErrDrafterStreamed is ErrDrafterReleased's counterpart for a drafter opened to stream its matrices to a resident
// (OpenDFlashDrafter): it holds none, so it cannot draft on the host.
var ErrDrafterStreamed = errors.New("decoder: the drafter was opened to stream its matrices to a resident (OpenDFlashDrafter); load it with LoadDFlashDrafter to draft on the host")

// weightsErr says why host drafting cannot read the matrices, or nil when it can.
func (d *blockTrunk) weightsErr() error {
	switch {
	case d.released:
		return ErrDrafterReleased
	case d.rows != nil:
		return ErrDrafterStreamed
	}
	return nil
}

// mustHoldWeights panics with weightsErr's error, for the host drafting methods and matrix accessors with no error result.
func (d *blockTrunk) mustHoldWeights() {
	if err := d.weightsErr(); err != nil {
		panic(err)
	}
}

// mustHoldNorms panics with ErrDrafterReleased on a released drafter: a streamed one holds its norms.
func (d *blockTrunk) mustHoldNorms() {
	if d.released {
		panic(ErrDrafterReleased)
	}
}

// Streamed reports whether the drafter was opened to stream its matrices (StreamedDrafterWeights).
func (d *blockTrunk) Streamed() bool { return d.rows != nil }

// DrafterFCRows is fc's row reader on a streamed drafter; it panics on any other.
func (d *blockTrunk) DrafterFCRows() DrafterRows { return d.rowsOf(&d.fc) }

// DrafterLayerRows is layer i's matrices as row readers, with its norms, on a streamed drafter; it panics on any other.
func (d *blockTrunk) DrafterLayerRows(i int) DrafterLayerRows {
	d.mustStream()
	l := &d.layers[i]
	return DrafterLayerRows{
		Q: d.rowsOf(&l.q), K: d.rowsOf(&l.k), V: d.rowsOf(&l.v), O: d.rowsOf(&l.o),
		Gate: d.rowsOf(&l.gate), Up: d.rowsOf(&l.up), Down: d.rowsOf(&l.down),
		QNorm: l.qNorm, KNorm: l.kNorm, InputNorm: l.inputNorm, PostAttnNorm: l.postAttnNorm,
	}
}

func (d *blockTrunk) rowsOf(w *linalg.WeightMat) DrafterRows {
	d.mustStream()
	return d.rows[w]
}

// mustStream panics unless the drafter was opened to stream and is not released.
func (d *blockTrunk) mustStream() {
	d.mustHoldNorms()
	if d.rows == nil {
		panic("decoder: a row reader asked of a drafter not opened to stream (OpenDFlashDrafter)")
	}
}

// stFile is a safetensors file's tensor byte ranges, read with ReadAt rather than through a mapping: a row read decodes straight
// into its destination, so streaming a matrix allocates nothing per block beyond a pooled byte buffer and maps no page of it.
// Safe for concurrent reads (os.File.ReadAt); a read after Close errors.
type stFile struct {
	f    *os.File
	base int64               // the payload's start: 8 + the header's length
	at   map[string][2]int64 // tensor name -> [start, end) within the payload
}

func openSTFile(path string) (*stFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*stFile, error) {
		f.Close()
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	var n [8]byte
	if _, err := f.ReadAt(n[:], 0); err != nil {
		return fail(err)
	}
	hl := int64(binary.LittleEndian.Uint64(n[:]))
	if hl <= 0 || hl > fi.Size()-8 {
		return fail(fmt.Errorf("safetensors header of %d bytes in a %d-byte file", hl, fi.Size()))
	}
	hdr := make([]byte, hl)
	if _, err := f.ReadAt(hdr, 8); err != nil {
		return fail(err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hdr, &raw); err != nil {
		return fail(err)
	}
	sf := &stFile{f: f, base: 8 + hl, at: make(map[string][2]int64, len(raw))}
	for name, m := range raw {
		if name == "__metadata__" {
			continue
		}
		var e struct {
			DataOffsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(m, &e); err != nil {
			return fail(fmt.Errorf("tensor %q: %w", name, err))
		}
		sf.at[name] = e.DataOffsets
	}
	return sf, nil
}

func (sf *stFile) Close() error { return sf.f.Close() }

// stRowBufs pools stRows' byte buffers, one per concurrent read.
var stRowBufs sync.Pool

// stRows is a row reader over tensor name [rows, cols] (F32, BF16 or F16), checked against st's view of the same file. Each read
// is one ReadAt of the block's bytes, at any byte offset, decoded into dst: neither the matrix nor its f32 widening is held, and
// the values are the bits LoadDFlashDrafter's matrices hold, widened.
func stRows(st *embed.SafetensorsFile, sf *stFile, name string, rows, cols int) (DrafterRows, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return DrafterRows{}, err
	}
	if !slices.Equal(t.Shape, []int{rows, cols}) {
		return DrafterRows{}, fmt.Errorf("tensor %q: shape %v, want [%d %d]", name, t.Shape, rows, cols)
	}
	esz := map[string]int{"F32": 4, "BF16": 2, "F16": 2}[t.DType]
	if esz == 0 {
		return DrafterRows{}, fmt.Errorf("tensor %q: dtype %s, want F32, BF16 or F16", name, t.DType)
	}
	rng, ok := sf.at[name]
	if !ok || rng[1]-rng[0] != int64(rows*cols*esz) {
		return DrafterRows{}, fmt.Errorf("tensor %q: byte range %v, want %d bytes", name, rng, rows*cols*esz)
	}
	dtype := t.DType
	return DrafterRows{Rows: rows, Cols: cols, Read: func(r0, n int, dst []float32) error {
		need := n * cols
		if r0 < 0 || n < 0 || r0+n > rows || len(dst) < need {
			return fmt.Errorf("tensor %q: rows %d..%d of %d into %d floats", name, r0, r0+n-1, rows, len(dst))
		}
		bp, _ := stRowBufs.Get().(*[]byte)
		if bp == nil || cap(*bp) < need*esz {
			b := make([]byte, need*esz)
			bp = &b
		}
		b := (*bp)[:need*esz]
		defer stRowBufs.Put(bp)
		if _, err := sf.f.ReadAt(b, sf.base+rng[0]+int64(r0*cols*esz)); err != nil {
			return fmt.Errorf("tensor %q: %w", name, err)
		}
		switch dtype {
		case "BF16":
			for i := range need {
				dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
			}
		case "F16":
			for i := range need {
				dst[i] = linalg.F16ToF32(binary.LittleEndian.Uint16(b[2*i:]))
			}
		default:
			for i := range need {
				dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
			}
		}
		return nil
	}}, nil
}

// releaseHostWeights drops the trunk's matrices and norms; the geometry stays.
func (d *blockTrunk) releaseHostWeights() {
	d.fc, d.hiddenNorm, d.finalNorm, d.rows = linalg.WeightMat{}, nil, nil, nil
	for i := range d.layers {
		d.layers[i] = dflashLayer{}
	}
	d.released = true
}

// HostWeightsReleased reports whether the host weights were released (ReleasableDrafterWeights).
func (d *blockTrunk) HostWeightsReleased() bool { return d.released }

type dflashLayer struct {
	q, k, v, o     linalg.WeightMat
	gate, up, down linalg.WeightMat
	qNorm, kNorm   []float32 // [headDim] — Qwen3 per-head RMSNorm
	inputNorm      []float32 // normalizes the BLOCK only (never the context — see attend)
	postAttnNorm   []float32
}

// BlockSize is the trained block width. The drafter proposes BlockSize-1 tokens: slot 0
// is the anchor (a token already committed), so it is embedded but never predicted.
func (d *DFlashDrafter) BlockSize() int { return d.blockSize }

// MaskTokenID is the id the untrained block slots are filled with.
func (d *DFlashDrafter) MaskTokenID() int { return d.maskTokenID }

// TargetLayerIDs are the target layers whose OUTPUTS feed `fc`. HF's reference indexes
// hidden_states[id+1] because hidden_states[0] is the embedding output, so these name
// layer outputs — pass them straight to Model.ForwardCapture, which uses the same
// after-layer-l convention.
func (d *DFlashDrafter) TargetLayerIDs() []int { return append([]int(nil), d.targetLayerIDs...) }

// Close releases the drafter's mmap'd weights.
func (d *DFlashDrafter) Close() error {
	var err error
	if d.sf != nil {
		err = d.sf.Close()
		d.sf = nil
	}
	if d.st != nil {
		st := d.st
		d.st = nil
		err = errors.Join(err, st.Close())
	}
	return err
}

// ReleaseHostWeights drops every host weight once a resident backend has uploaded its own copy (ResidentDrafterHost's
// AttachBlockDrafter): the trunk's, and an own head, embedding and d2t, which only host drafting reads. It closes the mapping,
// which nothing aliases afterwards. What a BlockSpec reads afterwards stays: BlockSize, MaskTokenID, TargetLayerIDs,
// DrafterGeometry and HasOwnHead. Host drafting on the drafter then fails with ErrDrafterReleased, and a backend's attach
// refuses it (ReleasableDrafterWeights). Not safe beside a call that reads the weights.
func (d *DFlashDrafter) ReleaseHostWeights() error {
	d.releaseHostWeights()
	d.lmHead, d.embed, d.d2t = linalg.WeightMat{}, linalg.WeightMat{}, nil
	return d.Close()
}

// dflashConfig is the subset of the checkpoint's config.json the loader needs.
type dflashConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	// RoPE comes in two spellings: flat rope_theta, or nested rope_parameters (as on DSpark checkpoints). Read both.
	RopeTheta      float64         `json:"rope_theta"`
	RopeParameters json.RawMessage `json:"rope_parameters"`
	// block_size is top-level on some checkpoints and nested in dflash_config on others. Read either.
	BlockSize int `json:"block_size"`
	DFlash    struct {
		BlockSize      int   `json:"block_size"`
		MaskTokenID    int   `json:"mask_token_id"`
		TargetLayerIDs []int `json:"target_layer_ids"`
		// DFlash 2 markers. This loader implements v1, and a v2 checkpoint is field-compatible with it (same trunk tensor
		// names, same config keys), so without these it would load silently. See the refusal in LoadDFlashDrafter.
		ConvKernelSize int `json:"conv_kernel_size"`
		ConvGroupSize  int `json:"conv_group_size"`
		SelectorRank   int `json:"selector_rank"`
		SelectorTopK   int `json:"selector_top_k"`
	} `json:"dflash_config"`

	// vLLM "speculators" dialect (v0.5), which poolside's Laguna drafters ship: a fourth config spelling for the same
	// model. It differs from z-lab's in three ways, all handled in LoadDFlashDrafter:
	//
	//   * the layer geometry is nested under transformer_layer_config (whose model_type says "llama" even though the
	//     layers carry Qwen3-style per-head q_norm/k_norm),
	//   * the taps are aux_hidden_state_layer_ids rather than target_layer_ids,
	//   * mask_token_id is top-level rather than inside dflash_config.
	//
	// DraftVocabSize marks the other structural difference: this drafter ships its own embed_tokens and a reduced-vocab
	// lm_head plus d2t/t2d, where z-lab's borrow the target's. See the own-head fields on DFlashDrafter.
	MaskTokenIDTop *int `json:"mask_token_id"`
	// The per-layer attention of the retrained drafters (z-lab's dflash/model.py, Qwen3DFlashAttention): a "sliding_attention"
	// layer is causal unless is_causal says otherwise, and windowed by sliding_window; a "full_attention" one is neither. Set,
	// is_causal applies to every layer.
	LayerTypes             []string        `json:"layer_types"`
	SlidingWindow          *int            `json:"sliding_window"`
	IsCausal               *bool           `json:"is_causal"`
	AuxHiddenStateLayerIDs []int           `json:"aux_hidden_state_layer_ids"`
	DraftVocabSize         int             `json:"draft_vocab_size"`
	VocabSize              int             `json:"vocab_size"`
	TransformerLayerConfig json.RawMessage `json:"transformer_layer_config"`
	SpeculatorsModelType   string          `json:"speculators_model_type"`
}

// LoadDFlashDrafter loads a DFlash drafter from dir (config.json + model.safetensors, the
// z-lab layout). The drafter's hidden size must equal the target's — it consumes the
// target's hidden states and embeddings directly, with no projection between them.
func LoadDFlashDrafter(dir string) (*DFlashDrafter, error) { return loadDFlashDrafter(dir, false) }

// OpenDFlashDrafter is LoadDFlashDrafter for a drafter a resident backend will hold: it reads the config and the safetensors
// header, checks every tensor as LoadDFlashDrafter does, and holds the norms and d2t, but no matrix and no mapping. Each matrix
// is a row reader over the file (stRows, StreamedDrafterWeights), which the backend's attach streams to the device, so the host
// never holds the drafter, however its file is aligned. Only a resident that streams drafter rows takes it
// (StreamingDrafterHost); it cannot draft on the host (ErrDrafterStreamed).
func OpenDFlashDrafter(dir string) (*DFlashDrafter, error) { return loadDFlashDrafter(dir, true) }

func loadDFlashDrafter(dir string, streamed bool) (*DFlashDrafter, error) {
	cfgBytes, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("dflash: read config: %w", err)
	}
	var c dflashConfig
	if err := json.Unmarshal(cfgBytes, &c); err != nil {
		return nil, fmt.Errorf("dflash: parse config: %w", err)
	}
	// Refuse DFlash 2. A v2 config carries every field this loader reads and most of its tensors are v1-shaped, so it would
	// load without error and silently drop the two-tap dynamic convolutions (layers.N.{attention,mlp}_conv.*) and the
	// candidate selector (candidate_selector.*). The convs sit before and after every attention and FFN sublayer, so
	// dropping them changes every layer's output. Verify is lossless, so the failure is not wrong tokens but a drafter
	// that drafts badly (lower acceptance, slower than v1) with no diagnostic. Detected from config rather than tensor
	// names because it is cheaper and is what the publisher controls; a v2 checkpoint that omitted these keys would still
	// be caught by the missing-tensor path. DFlash 2 support is unbuilt: docs/spec/08-dspark-dflash.md, section "DFlash 2".
	if c.DFlash.ConvKernelSize > 0 || c.DFlash.SelectorRank > 0 {
		return nil, fmt.Errorf("dflash: %s is a DFlash 2 checkpoint (conv_kernel_size=%d, "+
			"selector_rank=%d) and this loader implements v1 — it would load and silently ignore "+
			"the dynamic-conv and candidate-selector weights (23.8%% of the file on the released "+
			"Qwen3.8-27B pair), giving a drafter that verifies losslessly but drafts worse than v1. "+
			"DFlash 2 support is queued as P15", dir, c.DFlash.ConvKernelSize, c.DFlash.SelectorRank)
	}
	// speculators dialect: the layer geometry lives one level down. Decode it into
	// the SAME struct so every field below is read once, from one place, whichever
	// dialect the checkpoint used. Top-level keys already parsed (block_size,
	// mask_token_id, the tap list) are not repeated there, so this cannot clobber them.
	if len(c.TransformerLayerConfig) > 0 && c.HiddenSize == 0 {
		if err := json.Unmarshal(c.TransformerLayerConfig, &c); err != nil {
			return nil, fmt.Errorf("dflash: parse transformer_layer_config: %w", err)
		}
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.BlockSize == 0 {
		c.BlockSize = c.DFlash.BlockSize
	}
	// Taps and mask token in whichever spelling this checkpoint uses.
	if len(c.DFlash.TargetLayerIDs) == 0 {
		c.DFlash.TargetLayerIDs = c.AuxHiddenStateLayerIDs
	}
	if c.MaskTokenIDTop != nil && c.DFlash.MaskTokenID == 0 {
		c.DFlash.MaskTokenID = *c.MaskTokenIDTop
	}
	causal, window, err := dflashMasks(c)
	if err != nil {
		return nil, err
	}
	switch {
	case c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0:
		return nil, fmt.Errorf("dflash: bad dims (hidden=%d layers=%d heads=%d)", c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads)
	case c.BlockSize < 2:
		return nil, fmt.Errorf("dflash: block_size must be >= 2, got %d", c.BlockSize)
	case len(c.DFlash.TargetLayerIDs) == 0:
		return nil, fmt.Errorf("dflash: no taps — need dflash_config.target_layer_ids or aux_hidden_state_layer_ids")
	case c.RopeTheta <= 0 && len(c.RopeParameters) == 0:
		return nil, fmt.Errorf("dflash: no rope_theta and no rope_parameters")
	}

	theta := c.RopeTheta
	if len(c.RopeParameters) > 0 {
		spec, _, perr := parseRopeFlat(c.RopeParameters)
		if perr != nil {
			return nil, fmt.Errorf("dflash: rope_parameters: %w", perr)
		}
		theta = spec.base
	}
	if theta <= 0 {
		return nil, fmt.Errorf("dflash: rope_theta must be >0, got %v", theta)
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("dflash: open safetensors: %w", err)
	}
	loaded := false
	defer func() {
		if !loaded {
			st.Close()
		}
	}()

	h, hd := c.HiddenSize, c.HeadDim
	qDim, kvDim := c.NumAttentionHeads*hd, c.NumKeyValueHeads*hd
	d := &DFlashDrafter{
		hidden: h, nHeads: c.NumAttentionHeads, nKV: c.NumKeyValueHeads, headDim: hd,
		inter: c.IntermediateSize, normEps: c.RMSNormEps,
		invFreq:   computeInvFreq(theta, hd, nil),
		layers:    make([]dflashLayer, c.NumHiddenLayers),
		blockSize: c.BlockSize, maskTokenID: c.DFlash.MaskTokenID,
		targetLayerIDs: c.DFlash.TargetLayerIDs,
		st:             st,
	}
	d.layerCausal, d.layerWindow = causal, window
	if streamed {
		d.rows = map[*linalg.WeightMat]DrafterRows{}
		if d.sf, err = openSTFile(filepath.Join(dir, "model.safetensors")); err != nil {
			return nil, fmt.Errorf("dflash: %w", err)
		}
		defer func() {
			if !loaded {
				d.sf.Close()
			}
		}()
	}

	mat := func(dst *linalg.WeightMat, name string, rows, cols int) error {
		if streamed {
			r, err := stRows(st, d.sf, name, rows, cols)
			if err != nil {
				return fmt.Errorf("dflash: %w", err)
			}
			d.rows[dst] = r
			return nil
		}
		m, err := stIsland(st, name, rows, cols) // unquantized: a bf16 checkpoint's bits stay bf16, mapped (MEM11-bf16)
		if err != nil {
			return fmt.Errorf("dflash: %w", err)
		}
		*dst = m
		return nil
	}
	vec := func(dst *[]float32, name string, n int) error {
		v, err := st.TensorF32(name, n)
		if err != nil {
			return fmt.Errorf("dflash: %w", err)
		}
		if streamed {
			v = slices.Clone(v) // an F32 vector is a view of the mapping, which a streamed drafter closes once loaded
		}
		*dst = v
		return nil
	}

	if err := mat(&d.fc, "fc.weight", h, len(d.targetLayerIDs)*h); err != nil {
		return nil, err
	}
	if err := vec(&d.hiddenNorm, "hidden_norm.weight", h); err != nil {
		return nil, err
	}
	if err := vec(&d.finalNorm, "norm.weight", h); err != nil {
		return nil, err
	}
	// Own head, embedding and d2t, when this drafter ships them (draft_vocab_size), loaded together and validated against
	// each other: a mismatched d2t maps argmaxes into the wrong ids, which is invisible in output (drafting is lossless)
	// and shows up only as acceptance collapsing.
	if c.DraftVocabSize > 0 {
		d.draftVocab = c.DraftVocabSize
		if err := mat(&d.lmHead, "lm_head.weight", c.DraftVocabSize, h); err != nil {
			return nil, err
		}
		if c.VocabSize > 0 {
			if err := mat(&d.embed, "embed_tokens.weight", c.VocabSize, h); err != nil {
				return nil, err
			}
		}
		t, terr := st.Tensor("d2t")
		if terr != nil {
			return nil, fmt.Errorf("dflash: draft_vocab_size=%d but no d2t: %w", c.DraftVocabSize, terr)
		}
		i64, ierr := t.Int64s()
		if ierr != nil {
			return nil, fmt.Errorf("dflash: d2t as i64: %w", ierr)
		}
		if len(i64) != c.DraftVocabSize {
			return nil, fmt.Errorf("dflash: d2t len %d, want draft_vocab_size %d", len(i64), c.DraftVocabSize)
		}
		d.d2t = make([]int32, len(i64))
		for i, v := range i64 {
			d.d2t[i] = int32(v)
		}
		// Every mapped id must land inside the TARGET vocab; an out-of-range one would
		// index the target's embedding out of bounds at verify time.
		if c.VocabSize > 0 {
			for i, off := range d.d2t {
				if tid := i + int(off); tid < 0 || tid >= c.VocabSize {
					return nil, fmt.Errorf("dflash: d2t[%d] maps to target id %d, outside vocab %d", i, tid, c.VocabSize)
				}
			}
		}
	}
	for i := range d.layers {
		l := &d.layers[i]
		p := fmt.Sprintf("layers.%d.", i)
		for _, m := range []struct {
			dst        *linalg.WeightMat
			name       string
			rows, cols int
		}{
			{&l.q, p + "self_attn.q_proj.weight", qDim, h},
			{&l.k, p + "self_attn.k_proj.weight", kvDim, h},
			{&l.v, p + "self_attn.v_proj.weight", kvDim, h},
			{&l.o, p + "self_attn.o_proj.weight", h, qDim},
			{&l.gate, p + "mlp.gate_proj.weight", c.IntermediateSize, h},
			{&l.up, p + "mlp.up_proj.weight", c.IntermediateSize, h},
			{&l.down, p + "mlp.down_proj.weight", h, c.IntermediateSize},
		} {
			if err := mat(m.dst, m.name, m.rows, m.cols); err != nil {
				return nil, err
			}
		}
		for _, v := range []struct {
			dst  *[]float32
			name string
			n    int
		}{
			{&l.qNorm, p + "self_attn.q_norm.weight", hd},
			{&l.kNorm, p + "self_attn.k_norm.weight", hd},
			{&l.inputNorm, p + "input_layernorm.weight", h},
			{&l.postAttnNorm, p + "post_attention_layernorm.weight", h},
		} {
			if err := vec(v.dst, v.name, v.n); err != nil {
				return nil, err
			}
		}
	}
	if streamed {
		// Nothing the drafter keeps is a view of the mapping now (its rows read through d.sf), so close it: an opened drafter
		// holds one file and no mapping.
		if err := st.Close(); err != nil {
			return nil, fmt.Errorf("dflash: %w", err)
		}
		d.st = nil
	}
	loaded = true
	return d, nil
}

// FuseContext turns the target's captured hidden states into the drafter's context
// representation: fused[i] = hiddenNorm(fc · concat(h[l][i] for l in targetLayerIDs)).
// ctxCat is [ctxLen][len(targetLayerIDs)*hidden] — the ForwardCapture seam's output for
// each committed position, already concatenated in targetLayerIDs order.
//
// This is the ONLY place the target's hidden states enter, and the result is what every
// layer's K/V read. It is computed once per round, not per layer.
func (d *blockTrunk) FuseContext(be Backend, ctxCat [][]float32) ([][]float32, error) {
	if err := d.weightsErr(); err != nil {
		return nil, err
	}
	// fc's input width IS the tap count x hidden, so it validates the caller's concatenation
	// without the trunk needing to know which layers were tapped — that belongs to the drafter.
	want := d.fc.Cols()
	fused := make([][]float32, len(ctxCat))
	for i, row := range ctxCat {
		if len(row) != want {
			return nil, fmt.Errorf("block trunk: context row %d is %d wide, want %d (%d taps x %d)",
				i, len(row), want, want/d.hidden, d.hidden)
		}
		f := make([]float32, d.hidden)
		matmul(be, &d.fc, row, f, 1)
		rmsNorm(f, d.hiddenNorm, 1, d.hidden, d.normEps, false)
		fused[i] = f
	}
	return fused, nil
}

// DFlashContext caches the committed context's projected K/V per layer: K already RoPE'd at its absolute position, V
// neither roped nor normed. The context is projected once per position and read by every later round; without the
// cache each round re-projects the whole context in every layer, O(ctx x layers) of repeat work
// (BenchmarkDFlashTrunk).
//
// Append-only, plus TruncateTo for the speculative rollback: only committed tokens extend it, and a rejected draft's
// positions must come back off.
type DFlashContext struct {
	k, v [][][]float32 // [layer][pos][nKV*headDim]
}

// NewContext returns an empty per-layer context cache for this drafter.
func (d *blockTrunk) NewContext() *DFlashContext {
	return &DFlashContext{k: make([][][]float32, len(d.layers)), v: make([][][]float32, len(d.layers))}
}

// Len is the number of committed context positions cached.
func (c *DFlashContext) Len() int {
	if len(c.k) == 0 {
		return 0
	}
	return len(c.k[0])
}

// TruncateTo drops cached positions at index >= n — the rollback after a partial accept.
// The retained K was roped at its absolute position, so it stays valid after the trim.
func (c *DFlashContext) TruncateTo(n int) {
	for l := range c.k {
		if n < len(c.k[l]) {
			c.k[l] = c.k[l][:n]
			c.v[l] = c.v[l][:n]
		}
	}
}

// ExtendContext projects newly committed positions into the cache. fusedNew is
// FuseContext's output for those positions ONLY; they are assumed to sit immediately
// after the positions already cached, which is what fixes their RoPE offsets.
func (d *blockTrunk) ExtendContext(be Backend, ctx *DFlashContext, fusedNew [][]float32) {
	d.mustHoldWeights()
	start := ctx.Len()
	kvDim := d.nKV * d.headDim
	for li := range d.layers {
		l := &d.layers[li]
		for i, row := range fusedNew {
			k := make([]float32, kvDim)
			v := make([]float32, kvDim)
			matmul(be, &l.k, row, k, 1)
			matmul(be, &l.v, row, v, 1)
			rmsNorm(k, l.kNorm, d.nKV, d.headDim, d.normEps, false)
			applyRoPE(k, d.nKV, d.headDim, start+i, d.invFreq, 1)
			ctx.k[li] = append(ctx.k[li], k)
			ctx.v[li] = append(ctx.v[li], v)
		}
	}
}

// DraftBlockCtx runs the trunk over one block against a cached context and returns the
// final-normed hidden states, [blockSize][hidden]. This is the form production should
// use; DraftBlock is the uncached convenience wrapper.
func (d *blockTrunk) DraftBlockCtx(be Backend, ctx *DFlashContext, blockIn [][]float32) ([][]float32, error) {
	if err := d.weightsErr(); err != nil {
		return nil, err
	}
	// The block's WIDTH is whatever the caller passes — the trained width is the drafter's
	// property, not the trunk's, and the enclosing type checks it.
	if len(blockIn) == 0 {
		return nil, fmt.Errorf("block trunk: empty block")
	}
	h := make([][]float32, len(blockIn))
	for i, row := range blockIn {
		if len(row) != d.hidden {
			return nil, fmt.Errorf("block trunk: block row %d is %d wide, want %d", i, len(row), d.hidden)
		}
		h[i] = append([]float32(nil), row...)
	}
	for i := range d.layers {
		d.layer(be, i, ctx.k[i], ctx.v[i], h)
	}
	out := make([][]float32, len(h))
	for i, row := range h {
		r := append([]float32(nil), row...)
		rmsNorm(r, d.finalNorm, 1, d.hidden, d.normEps, false)
		out[i] = r
	}
	return out, nil
}

// DraftBlock runs the trunk over one block and returns its final-normed hidden states,
// [blockSize][hidden]. blockIn is the TARGET's embedding of the block's token ids (slot 0
// the anchor, the rest MASK); fused is FuseContext's output for the committed context.
// The caller applies the target's LM head to rows 1.. to get the drafted logits.
//
// Uncached: it projects the whole context every call. Convenient for a one-shot round (and
// for the parity gate, which is handed the reference's fused context directly), but a
// generation loop should hold a DFlashContext and call DraftBlockCtx.
func (d *blockTrunk) DraftBlock(be Backend, fused, blockIn [][]float32) ([][]float32, error) {
	if err := d.weightsErr(); err != nil {
		return nil, err
	}
	ctx := d.NewContext()
	d.ExtendContext(be, ctx, fused)
	return d.DraftBlockCtx(be, ctx, blockIn)
}

// DrafterHeadLogits applies the TARGET's LM head to one already-normed hidden row and
// returns next-token logits over the target vocabulary. It is the output end that a
// drafter shipping no `lm_head` of its own (DFlash) borrows from the target.
//
// It deliberately does NOT apply the target's final norm — the drafter has already
// applied its OWN `norm`, and running both would be a second normalization the reference
// does not do. That is the only difference from logitsFromHidden, and it is why this
// cannot just call it. The post-head transforms (Gemma softcap, Granite logit scale) ARE
// applied, mirroring the reference's compute_logits.
//
// Declared here rather than in model.go on purpose: model.go is in the parity manifest's core set, and adding a
// drafter-only accessor there would re-stale every family's deps_hash for a function no existing forward calls.
func (m *Model) DrafterHeadLogits(h []float32) []float32 {
	arch := m.w.arch
	logits := make([]float32, arch.VocabSize)
	if arch.TiedLMHead {
		matmul(m.be, &m.w.Embed, h, logits, 1)
	} else {
		matmul(m.be, &m.w.LMHead, h, logits, 1)
	}
	if arch.FinalLogitSoftcap > 0 {
		softcapParallel(logits, float32(arch.FinalLogitSoftcap))
	}
	if arch.LogitScale != 0 && arch.LogitScale != 1 {
		inv := float32(1 / arch.LogitScale)
		for i := range logits {
			logits[i] *= inv
		}
	}
	return logits
}

// DrafterEmbedBlock fills the drafter's block input with the TARGET's token embeddings —
// the input end DFlash borrows. ids must be blockSize long (slot 0 the anchor, the rest
// the mask token).
func (m *Model) DrafterEmbedBlock(ids []int) [][]float32 {
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		rows[i] = make([]float32, m.w.arch.HiddenDim)
		m.embedToken(id, rows[i])
	}
	return rows
}

// layer runs one trunk layer in place over the block rows.
//
// The asymmetry worth naming: `input_layernorm` normalizes the BLOCK only. The context's
// K/V are projected from the fused context RAW (in ExtendContext) — the reference passes
// `target_hidden` straight into k_proj/v_proj while only `hidden_states` goes through the
// norm. Norming both would be the natural-looking port and would be wrong.
func (d *blockTrunk) layer(be Backend, li int, ctxK, ctxV [][]float32, h [][]float32) {
	l := &d.layers[li]
	hid, hd := d.hidden, d.headDim
	qDim, kvDim := d.nHeads*hd, d.nKV*hd
	nBlk, nCtx := len(h), len(ctxK)
	nKeys := nCtx + nBlk

	// Block rows, normed, are the attention input for Q and for the block's own K/V.
	xb := make([][]float32, nBlk)
	for i, row := range h {
		x := append([]float32(nil), row...)
		rmsNorm(x, l.inputNorm, 1, hid, d.normEps, false)
		xb[i] = x
	}

	// Keys/values: the context's come straight from the cache (projected and roped once,
	// when those positions were committed); only the BLOCK's are computed here. RoPE
	// positions are ABSOLUTE over [context ‖ block], which is what makes the block's start
	// at nCtx and q's likewise.
	keys := make([][]float32, nKeys)
	vals := make([][]float32, nKeys)
	copy(keys, ctxK)
	copy(vals, ctxV)
	for j := nCtx; j < nKeys; j++ {
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		matmul(be, &l.k, xb[j-nCtx], k, 1)
		matmul(be, &l.v, xb[j-nCtx], v, 1)
		rmsNorm(k, l.kNorm, d.nKV, hd, d.normEps, false) // per-head, Qwen3 style
		applyRoPE(k, d.nKV, hd, j, d.invFreq, 1)         // V is NOT roped and NOT normed
		keys[j], vals[j] = k, v
	}

	scale := 1 / math.Sqrt(float64(hd))
	scores := make([]float64, nKeys)
	ctxRow := make([]float32, qDim)
	attnOut := make([]float32, hid)
	for i := range nBlk {
		q := make([]float32, qDim)
		matmul(be, &l.q, xb[i], q, 1)
		rmsNorm(q, l.qNorm, d.nHeads, hd, d.normEps, false)
		applyRoPE(q, d.nHeads, hd, nCtx+i, d.invFreq, 1)

		// The layer's mask (keyVisible): a full layer attends every context key and every block key, positions after itself
		// included; a causal or windowed one, the reference's subset.
		qPos := nCtx + i
		for hI := range d.nHeads {
			kvH := hI / (d.nHeads / d.nKV) // GQA broadcast
			qo, ko := hI*hd, kvH*hd
			maxS := math.Inf(-1)
			for j := range nKeys {
				if !d.keyVisible(li, qPos, j) {
					scores[j] = math.Inf(-1)
					continue
				}
				var dot float64
				kr := keys[j][ko : ko+hd]
				for t, qv := range q[qo : qo+hd] {
					dot += float64(qv) * float64(kr[t])
				}
				s := dot * scale
				scores[j] = s
				if s > maxS {
					maxS = s
				}
			}
			var sum float64
			for j := range nKeys {
				e := math.Exp(scores[j] - maxS)
				scores[j] = e
				sum += e
			}
			for t := range hd {
				var acc float64
				for j := range nKeys {
					acc += scores[j] * float64(vals[j][ko+t])
				}
				ctxRow[qo+t] = float32(acc / sum)
			}
		}
		matmul(be, &l.o, ctxRow, attnOut, 1)
		for t := range hid {
			h[i][t] += attnOut[t]
		}
	}

	// SwiGLU MLP over the post-attention residual.
	gate := make([]float32, d.inter)
	up := make([]float32, d.inter)
	down := make([]float32, hid)
	for i := range nBlk {
		x := append([]float32(nil), h[i]...)
		rmsNorm(x, l.postAttnNorm, 1, hid, d.normEps, false)
		matmul(be, &l.gate, x, gate, 1)
		matmul(be, &l.up, x, up, 1)
		for t := range gate {
			gate[t] = silu(gate[t]) * up[t]
		}
		matmul(be, &l.down, gate, down, 1)
		for t := range hid {
			h[i][t] += down[t]
		}
	}
}

// dflashMasks is each layer's mask from the config, as z-lab's reference reads it (Qwen3DFlashAttention.__init__): nil, nil for a
// drafter that names no layer types (every layer full attention, the original DFlash).
func dflashMasks(c dflashConfig) ([]bool, []int, error) {
	if len(c.LayerTypes) == 0 {
		if c.IsCausal != nil && *c.IsCausal {
			return nil, nil, fmt.Errorf("dflash: is_causal set with no layer_types")
		}
		return nil, nil, nil
	}
	if len(c.LayerTypes) != c.NumHiddenLayers {
		return nil, nil, fmt.Errorf("dflash: %d layer_types for %d layers", len(c.LayerTypes), c.NumHiddenLayers)
	}
	causal, window := make([]bool, len(c.LayerTypes)), make([]int, len(c.LayerTypes))
	for i, t := range c.LayerTypes {
		switch t {
		case "full_attention":
		case "sliding_attention":
			causal[i] = true
			if c.SlidingWindow == nil || *c.SlidingWindow <= 0 {
				return nil, nil, fmt.Errorf("dflash: layer %d is sliding_attention with no positive sliding_window", i)
			}
			window[i] = *c.SlidingWindow
		default:
			return nil, nil, fmt.Errorf("dflash: layer %d has layer type %q", i, t)
		}
		if c.IsCausal != nil {
			causal[i] = *c.IsCausal
		}
	}
	return causal, window, nil
}

// keyVisible reports whether layer li's block row at absolute position q sees key position k (z-lab's _attention_mask).
func (d *blockTrunk) keyVisible(li, q, k int) bool {
	if d.layerCausal == nil {
		return true
	}
	if d.layerCausal[li] && k > q {
		return false
	}
	if w := d.layerWindow[li]; w > 0 && (q-k >= w || (!d.layerCausal[li] && k-q >= w)) {
		return false
	}
	return true
}

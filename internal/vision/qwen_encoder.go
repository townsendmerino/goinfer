package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// Qwen2.5-VL vision tower — aikit's second ViT family (after SigLIP / encoder.go),
// the Qwen2.5-VL `.visual` submodule as a pure-Go fp32 forward. Where SigLIP is
// fixed-resolution (896×896 → 256 tokens, learned absolute pos, LayerNorm,
// gelu-tanh MLP), this is DYNAMIC-resolution: pre-flattened patches + grid_thw,
// 2D rotary, RMSNorm, windowed + full attention, a gated SiLU MLP, and a
// spatial-merge patch merger. Parity is cosine vs the HF
// Qwen2_5_VisionTransformerPretrainedModel golden (scripts/oracle/pin_qwen25vl_vision.py),
// gated in two stages: the ViT pre-merge hidden and the merged image features.
//
// Forward takes pre-patchified input (the goinfer P5.3 preprocessor does
// image→pixel_values+grid_thw via smart-resize upstream), not a CHW image — so the
// encoder is fed pixel_values [n_patches, patch_dim] + per-image (t,h,w) grids.
//
// Added ADDITIVELY: SigLIP's Encoder/LoadEncoder are untouched. The linalg.WeightMat W8A8
// wrapper is reused for the projections (the patch-embed matmul stays f32); the
// resident-GPU seam is a follow-on (the fp32 CPU path is the v1 deliverable).

// QwenEncoderConfig mirrors the HF Qwen2_5_VLVisionConfig fields the forward needs.
type QwenEncoderConfig struct {
	Depth               int    `json:"depth"`
	HiddenSize          int    `json:"hidden_size"`
	IntermediateSize    int    `json:"intermediate_size"`
	NumHeads            int    `json:"num_heads"`
	InChans             int    `json:"in_chans"`
	PatchSize           int    `json:"patch_size"`
	SpatialMergeSize    int    `json:"spatial_merge_size"`
	TemporalPatchSize   int    `json:"temporal_patch_size"`
	OutHiddenSize       int    `json:"out_hidden_size"`
	WindowSize          int    `json:"window_size"`
	FullattBlockIndexes []int  `json:"fullatt_block_indexes"`
	HiddenAct           string `json:"hidden_act"`
}

// validate rejects a config whose dimensions would divide-by-zero or
// mis-partition at load/Forward (H8): head_dim = hidden/num_heads, the
// merge-unit groups = n_patches/spatial_merge_size², and the window grid
// vmws = window_size/spatial_merge_size/patch_size all ÷0 on an absent field.
// Called after config parse + defaults, before any dimension is used.
func (c QwenEncoderConfig) validate() error {
	switch {
	case c.HiddenSize <= 0:
		return fmt.Errorf("hidden_size must be > 0, got %d", c.HiddenSize)
	case c.IntermediateSize <= 0:
		return fmt.Errorf("intermediate_size must be > 0, got %d", c.IntermediateSize)
	case c.Depth < 0:
		return fmt.Errorf("depth must be >= 0, got %d", c.Depth)
	case c.NumHeads <= 0:
		return fmt.Errorf("num_heads must be > 0, got %d", c.NumHeads)
	case c.HiddenSize%c.NumHeads != 0:
		return fmt.Errorf("hidden_size %d not divisible by num_heads %d", c.HiddenSize, c.NumHeads)
	case (c.HiddenSize/c.NumHeads)%4 != 0:
		// The rotary path derives rdim = head_dim/2 and len(inv_freq) =
		// head_dim/4; a head_dim not divisible by 4 makes rdim/inv_freq degenerate
		// and computeViTHidden divides by zero at first Forward. (num_heads > 0 and
		// hidden%num_heads == 0 are guaranteed above.)
		return fmt.Errorf("head_dim %d (hidden_size/num_heads) must be divisible by 4 for rotary", c.HiddenSize/c.NumHeads)
	case c.InChans <= 0:
		return fmt.Errorf("in_chans must be > 0, got %d", c.InChans)
	case c.PatchSize <= 0:
		return fmt.Errorf("patch_size must be > 0, got %d", c.PatchSize)
	case c.TemporalPatchSize <= 0:
		// patch_dim = in_chans·temporal·patch² feeds the patch-embed matmul; a 0
		// makes it zero-wide → an all-zero embedding, silently.
		return fmt.Errorf("temporal_patch_size must be > 0, got %d", c.TemporalPatchSize)
	case c.SpatialMergeSize <= 0:
		return fmt.Errorf("spatial_merge_size must be > 0, got %d", c.SpatialMergeSize)
	case c.OutHiddenSize <= 0:
		return fmt.Errorf("out_hidden_size must be > 0, got %d", c.OutHiddenSize)
	case c.WindowSize < c.SpatialMergeSize*c.PatchSize:
		return fmt.Errorf("window_size %d must be >= spatial_merge_size*patch_size = %d (else the window grid divides by zero)",
			c.WindowSize, c.SpatialMergeSize*c.PatchSize)
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		// mlp() hardcodes SiLU; a checkpoint declaring gelu/quick_gelu would
		// silently run the wrong activation (the H8 "known activation" check).
		return fmt.Errorf("hidden_act %q unsupported (silu only)", c.HiddenAct)
	}
	return nil
}

type qwenBlock struct {
	norm1w     []float32        // RMSNorm (weight only)
	qkvw       linalg.WeightMat // [3*hidden, hidden] fused
	qkvb       []float32        // [3*hidden]
	projw      linalg.WeightMat // [hidden, hidden]
	projb      []float32
	norm2w     []float32        // RMSNorm
	gatew, upw linalg.WeightMat // [inter, hidden]
	gateb, upb []float32
	downw      linalg.WeightMat // [hidden, inter]
	downb      []float32
}

// QwenVisionEncoder is a loaded Qwen2.5-VL vision tower (dynamic resolution).
type QwenVisionEncoder struct {
	Cfg        QwenEncoderConfig
	patchW     []float32 // [hidden, patch_dim] (Conv3d weight flattened, kept f32)
	blocks     []qwenBlock
	mergerLNw  []float32        // merger.ln_q RMSNorm weight [hidden]
	merger0w   linalg.WeightMat // merger.mlp.0 [hidden*merge², hidden*merge²]
	merger0b   []float32
	merger2w   linalg.WeightMat // merger.mlp.2 [out_hidden, hidden*merge²]
	merger2b   []float32
	rotInvFreq []float32 // head_dim/4 rotary frequencies

	resident QwenResidentEncoder // device-resident ViT (EnableResident); nil = CPU path
	rel      *blockRelease[QwenVisionEncoder]
	maps     towerMaps // the checkpoint mappings its aliased weights read (aliasQMat); Close releases them
}

const qwenRotaryTheta = 10000.0 // Qwen2_5_VisionRotaryEmbedding default theta

// LoadQwenVisionEncoder reads a Qwen2.5-VL checkpoint (config.json vision_config +
// safetensors) and returns a ready encoder. Weights are copied out, so the
// safetensors file is closed before return (no retained mmap). quant wraps the
// projections as int8 W8A8 (the patch-embed matmul stays f32); the fp32 parity gate
// runs quant=false.
func LoadQwenVisionEncoder(dir string, quant bool) (*QwenVisionEncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	// The vision config is nested under "vision_config" in a real VL checkpoint; a
	// stripped tower could carry it flat — prefer the nested one when present.
	var wrap struct {
		QwenEncoderConfig
		VisionConfig *QwenEncoderConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	cfg := wrap.QwenEncoderConfig
	if wrap.VisionConfig != nil {
		cfg = *wrap.VisionConfig
	}
	if cfg.InChans == 0 {
		cfg.InChans = 3
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("vision: %w", err)
	}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("vision: open safetensors: %w", err)
	}
	aliased, kept := false, false // a weight aliases st; the loaded tower keeps st open
	defer func() {
		if !kept {
			st.Close()
		}
	}()

	e := &QwenVisionEncoder{Cfg: cfg}
	// "visual." in the tiny checkpoint, "model.visual." inside a full HF VL
	// checkpoint (where the tower lives alongside the language model shards).
	pfx := qwenTensorPrefix(st)
	// get reads a tensor and, when want dims are given, shape-checks it (H7):
	// otherwise a mismatched/hostile checkpoint panics deep in QuantizeRowsInt8 or
	// MatmulBT at load/Forward instead of returning an error. Shapes follow HF
	// Qwen2.5-VL (fused-QKV/Linear weights [out,in], RMSNorms [hidden], 1-D
	// biases); patch_embed.proj is a 5-D Conv3d and is left unchecked. The parity
	// test (testdata/qwen25vl-vision-tiny) is the gate.
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = st.TensorF32(pfx+name, want...)
		if err != nil {
			return nil
		}
		return append([]float32(nil), v...) // copy out so st can close
	}
	hidden, inter := cfg.HiddenSize, cfg.IntermediateSize
	qm := func(name string, rows, cols int) linalg.WeightMat {
		if err != nil {
			return linalg.WeightMat{}
		}
		if w, ok, aerr := aliasQMat(st, pfx+name, rows, cols, quant); aerr != nil || ok {
			err, aliased = aerr, aliased || ok
			return w
		}
		w := get(name, rows, cols)
		if err != nil {
			return linalg.WeightMat{}
		}
		return newQMat(w, rows, cols, quant)
	}
	// Conv3d weight [hidden, in_chans, temporal, patch, patch] — shape-checked
	// like every other tensor (H7), so a mismatch is a clean load-time error, not
	// a later matmul panic / silent prefix.
	e.patchW = get("patch_embed.proj.weight", hidden, cfg.InChans, cfg.TemporalPatchSize, cfg.PatchSize, cfg.PatchSize)
	e.blocks = make([]qwenBlock, cfg.Depth)
	for i := range e.blocks {
		p := fmt.Sprintf("blocks.%d.", i)
		b := &e.blocks[i]
		b.norm1w = get(p+"norm1.weight", hidden)
		b.qkvw, b.qkvb = qm(p+"attn.qkv.weight", 3*hidden, hidden), get(p+"attn.qkv.bias", 3*hidden)
		b.projw, b.projb = qm(p+"attn.proj.weight", hidden, hidden), get(p+"attn.proj.bias", hidden)
		b.norm2w = get(p+"norm2.weight", hidden)
		b.gatew, b.gateb = qm(p+"mlp.gate_proj.weight", inter, hidden), get(p+"mlp.gate_proj.bias", inter)
		b.upw, b.upb = qm(p+"mlp.up_proj.weight", inter, hidden), get(p+"mlp.up_proj.bias", inter)
		b.downw, b.downb = qm(p+"mlp.down_proj.weight", hidden, inter), get(p+"mlp.down_proj.bias", hidden)
	}
	mh := hidden * cfg.SpatialMergeSize * cfg.SpatialMergeSize
	e.mergerLNw = get("merger.ln_q.weight", hidden)
	e.merger0w, e.merger0b = qm("merger.mlp.0.weight", mh, mh), get("merger.mlp.0.bias", mh)
	e.merger2w, e.merger2b = qm("merger.mlp.2.weight", cfg.OutHiddenSize, mh), get("merger.mlp.2.bias", cfg.OutHiddenSize)
	if err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", err)
	}

	// Rotary inv_freq over head_dim/2 (the Qwen2_5_VisionRotaryEmbedding dim) —
	// inv_freq = 1/theta^(arange(0,dim,2)/dim), i.e. head_dim/4 frequencies.
	headDim := hidden / cfg.NumHeads
	rdim := headDim / 2
	e.rotInvFreq = make([]float32, rdim/2)
	for i := range e.rotInvFreq {
		e.rotInvFreq[i] = float32(1.0 / math.Pow(qwenRotaryTheta, float64(2*i)/float64(rdim)))
	}
	e.rel = newBlockRelease(func() (*QwenVisionEncoder, error) { return LoadQwenVisionEncoder(dir, quant) })
	if aliased {
		e.maps.keep(st)
		kept = true
	}
	return e, nil
}

// qwenTensorPrefix reports the namespace the tower is nested under: "visual." for
// the tiny saved checkpoint, "model.visual." inside a full HF VL safetensors.
func qwenTensorPrefix(st *embed.SafetensorsFile) string {
	for _, pfx := range []string{"visual.", "model.visual."} {
		if _, err := st.Tensor(pfx + "patch_embed.proj.weight"); err == nil {
			return pfx
		}
	}
	return "visual."
}

// Forward runs the ViT + merger on pre-patchified pixel_values [n_patches, patch_dim]
// (patch_dim = in_chans*temporal*patch*patch) with per-image grids (t,h,w in patch
// units, h/w multiples of spatial_merge_size). It returns the merged image
// embeddings [n_merged, out_hidden_size] in ORIGINAL patch order — the embeddings
// that replace the decoder's <image> placeholders. n_merged = Σ t*h*w / merge².
func (e *QwenVisionEncoder) Forward(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	hid, err := e.ForwardViT(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	return e.merge(hid, gridTHW), nil
}

// computeViTHidden runs only the transformer blocks (no merger), returning the
// pre-merge hidden state [n_patches, hidden] in ORIGINAL patch order — the stage
// the parity gate checks against HF's last_hidden_state. (HF returns
// last_hidden_state in WINDOW order; for a self-contained gate we de-window here
// so the caller sees original order. The encoder_test compares against an
// order-matched golden.)
//
// Named distinctly from the exported ForwardViT below (rather than the
// lowercase/uppercase pair this used to be) so a case-insensitive search or
// skim can't conflate the CPU implementation with the public dispatcher that
// picks between it and the device-resident path.
func (e *QwenVisionEncoder) computeViTHidden(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	if err := e.ensureBlocks(); err != nil {
		return nil, err
	}
	c := e.Cfg
	merge := c.SpatialMergeSize
	mergeUnit := merge * merge
	hidden := c.HiddenSize
	patchDim := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize

	nPatches := 0
	for _, g := range gridTHW {
		// Per-image validation: h and w must each be divisible by
		// spatial_merge_size. The global nPatches%merge² check below is not
		// sufficient — e.g. {1,3,4} with merge 2 passes it (12%4==0) but h=3
		// isn't divisible, so windowIndex/rotaryFreqs emit fewer than `groups`
		// entries and winIdx[g] indexes OOB.
		if g[0] <= 0 || g[1] <= 0 || g[2] <= 0 {
			// <= 0, not < 0: a zero dim yields nPatches==0, which passes both
			// checks below (0 != 0 is false; 0 % merge² == 0) and then divides by
			// zero at `len(freqs)/nPatches`. Audit #4.
			return nil, fmt.Errorf("vision: non-positive grid dim in %v", g)
		}
		if g[1]%merge != 0 || g[2]%merge != 0 {
			return nil, fmt.Errorf("vision: grid %v h/w not divisible by spatial_merge_size %d", g, merge)
		}
		nPatches += g[0] * g[1] * g[2]
	}
	if nPatches == 0 {
		// Empty gridTHW (e.g. Forward(nil, nil) — a request with zero images)
		// reaches here with nPatches still 0; error rather than divide by zero.
		return nil, fmt.Errorf("vision: empty gridTHW (no images/patches)")
	}
	if len(pixelValues) != nPatches*patchDim {
		return nil, fmt.Errorf("vision: pixel_values len %d, want %d (%d patches × %d)", len(pixelValues), nPatches*patchDim, nPatches, patchDim)
	}
	if nPatches%mergeUnit != 0 {
		return nil, fmt.Errorf("vision: n_patches %d not a multiple of merge² %d", nPatches, mergeUnit)
	}

	// 1. patch embed: h[n,hidden] = pixel_values[n,patch_dim] · patchWᵀ (no bias).
	h := make([]float32, nPatches*hidden)
	linalg.MatmulBT(pixelValues, e.patchW, h, nPatches, patchDim, hidden)

	// 2. rotary freqs per patch (head_dim/2 each) from the (h_idx,w_idx) grid coords.
	freqs := e.rotaryFreqs(gridTHW) // [nPatches][rdim], original patch order

	// 3. window reorder (at merge-unit granularity) + the two cu_seqlens.
	winIdx, cuWin := e.windowIndex(gridTHW)
	cuFull := cuSeqlensFull(gridTHW)
	groups := nPatches / mergeUnit

	// reorder hidden + freqs into window order, grouping merge_unit patches.
	rdim := len(freqs) / nPatches
	hWin := make([]float32, nPatches*hidden)
	fWin := make([]float32, nPatches*rdim)
	for g := range groups {
		src := winIdx[g]
		for u := range mergeUnit {
			dp, sp := (g*mergeUnit+u)*hidden, (src*mergeUnit+u)*hidden
			copy(hWin[dp:dp+hidden], h[sp:sp+hidden])
			df, sf := (g*mergeUnit+u)*rdim, (src*mergeUnit+u)*rdim
			copy(fWin[df:df+rdim], freqs[sf:sf+rdim])
		}
	}

	// precompute cos/sin per patch over the full head_dim (emb = cat(freqs,freqs)).
	headDim := hidden / c.NumHeads
	cos := make([]float32, nPatches*headDim)
	sin := make([]float32, nPatches*headDim)
	for i := 0; i < nPatches; i++ {
		fr := fWin[i*rdim : i*rdim+rdim]
		for d := range headDim {
			f := float64(fr[d%rdim]) // emb[d]=freqs[d] (d<rdim), freqs[d-rdim] (d≥rdim)
			cos[i*headDim+d] = float32(math.Cos(f))
			sin[i*headDim+d] = float32(math.Sin(f))
		}
	}

	// 4. blocks (pre-norm residual). fullatt blocks attend per-image; others per-window.
	//
	// One arena for the whole tower (item 18); maxSeg covers both cu variants
	// because a full-attention block and a windowed block can each appear at
	// any depth.
	maxSeg := max(maxSegment(cuWin), maxSegment(cuFull))
	s := newQwenScratch(nPatches, hidden, c.IntermediateSize, headDim, maxSeg, c.NumHeads)
	att := s.att[:nPatches*hidden]
	o := s.o[:nPatches*hidden]
	mlpOut := s.mlpOut[:nPatches*hidden]
	n1 := s.n1[:nPatches*hidden]
	n2 := s.n2[:nPatches*hidden]
	for li := range e.blocks {
		cu := cuWin
		if e.isFullAtt(li) {
			cu = cuFull
		}
		b := &e.blocks[li]
		rmsNormInto(n1, hWin, b.norm1w, nPatches, hidden)
		if err := e.attentionInto(att, n1, b, nPatches, cos, sin, cu, s); err != nil {
			return nil, err
		}
		b.projw.MatmulBT(att, o, nPatches)
		addBias(o, b.projb, nPatches, hidden)
		addResidual(hWin, o)
		rmsNormInto(n2, hWin, b.norm2w, nPatches, hidden)
		e.mlpInto(mlpOut, n2, b, nPatches, s)
		addResidual(hWin, mlpOut)
	}

	// de-window back to original patch order (merge-unit granularity).
	out := make([]float32, nPatches*hidden)
	for g := range groups {
		dst := winIdx[g]
		for u := range mergeUnit {
			dp, sp := (dst*mergeUnit+u)*hidden, (g*mergeUnit+u)*hidden
			copy(out[dp:dp+hidden], hWin[sp:sp+hidden])
		}
	}
	return out, nil
}

// ForwardViT exposes the pre-merge hidden state for stage-isolated parity tests.
func (e *QwenVisionEncoder) ForwardViT(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	if err := e.maps.check(); err != nil {
		return nil, err
	}
	// Device-resident path (EnableResident): the whole ViT runs on the GPU and
	// returns the same de-windowed, original-order hidden state. The merger stays on
	// the CPU (see QwenResidentEncoder's doc for why).
	if e.resident != nil {
		return e.resident.ForwardViT(pixelValues, gridTHW)
	}
	return e.computeViTHidden(pixelValues, gridTHW)
}

// merge runs the patch merger on the ViT hidden (original patch order):
// RMSNorm(ln_q) → reshape merge² patches into one hidden*merge² vector → mlp.0 →
// GELU(erf) → mlp.2. Output [n_merged, out_hidden], one row per merge-unit group.
func (e *QwenVisionEncoder) merge(hidden []float32, gridTHW [][3]int) []float32 {
	c := e.Cfg
	H := c.HiddenSize
	mergeUnit := c.SpatialMergeSize * c.SpatialMergeSize
	mh := H * mergeUnit
	nPatches := len(hidden) / H
	groups := nPatches / mergeUnit

	// ln_q over hidden, then the [groups, mh] view falls out for free (contiguous).
	nrm := rmsNorm(hidden, e.mergerLNw, nPatches, H)
	mid := make([]float32, groups*mh)
	e.merger0w.MatmulBT(nrm, mid, groups)
	addBias(mid, e.merger0b, groups, mh)
	geluErf(mid)
	out := make([]float32, groups*c.OutHiddenSize)
	e.merger2w.MatmulBT(mid, out, groups)
	addBias(out, e.merger2b, groups, c.OutHiddenSize)
	return out
}

// attention runs bidirectional MHA within each cu_seqlens segment (window or full image), via
// linalg's fused schedule (P6, same treatment and same reasoning as encoder.go's attentionInto —
// see that function's and encHeadScratch's doc comments for the shared design). qkv is fused
// (reshape seq,3,heads,head_dim); 2D rotary is applied to q,k before attending. Segments run
// sequentially (there are only a handful per layer — image windows — so nesting parallelism
// inside them would just add contention); heads within one segment fan out across s.headPool.
//
// A decline from AttendTileFused is unreachable here for the same reason as SigLIP's: every
// headPool worker's FusedAttnScratch is sized via NewFusedAttnScratch(maxSeg, hd), and maxSeg is
// computed by the caller (computeViTHidden) as the longest run across BOTH cu_seqlens variants —
// every segment length n this function is ever called with satisfies n <= maxSeg, and
// FusedAttnScratch's buffers are prefix slices (Fits(n, hd) holds whenever n <= maxSeg).
func (e *QwenVisionEncoder) attentionInto(out, x []float32, b *qwenBlock, seq int, cos, sin []float32, cu []int, s *qwenScratch) error {
	return packedAttentionInto(out, x, b.qkvw, b.qkvb, e.Cfg.HiddenSize, e.Cfg.NumHeads, seq, cos, sin, cu, s)
}

// packedAttentionInto is attentionInto's body, factored out so the Qwen3.5+ tower (qwen3_encoder.go)
// shares the one fused-attention schedule instead of carrying a copy. Only the fused biased qkv
// projection enters it; the output projection, residual and norms stay with the caller. Moving the
// body did not change a single operation or its order.
func packedAttentionInto(out, x []float32, qkvw linalg.WeightMat, qkvb []float32, hidden, nH, seq int, cos, sin []float32, cu []int, s *qwenScratch) error {
	hd := hidden / nH

	qkv := s.qkv[:seq*3*hidden]
	qkvw.MatmulBT(x, qkv, seq)
	addBias(qkv, qkvb, seq, 3*hidden)
	// split: row layout is [3, nH, hd], so q/k/v are the three contiguous halves.
	q := s.q[:seq*hidden]
	k := s.k[:seq*hidden]
	v := s.v[:seq*hidden]
	for i := range seq {
		base := i * 3 * hidden
		copy(q[i*hidden:(i+1)*hidden], qkv[base:base+hidden])
		copy(k[i*hidden:(i+1)*hidden], qkv[base+hidden:base+2*hidden])
		copy(v[i*hidden:(i+1)*hidden], qkv[base+2*hidden:base+3*hidden])
	}
	// 2D rotary on q,k (NeoX rotate_half over the full head_dim).
	for i := range seq {
		co, si := cos[i*hd:i*hd+hd], sin[i*hd:i*hd+hd]
		for head := range nH {
			off := i*hidden + head*hd
			applyRotaryVision(q[off:off+hd], co, si)
			applyRotaryVision(k[off:off+hd], co, si)
		}
	}

	return attendPackedInto(out, q, k, v, hidden, nH, cu, s)
}

// attendPackedInto is the second half of packedAttentionInto — full bidirectional attention per
// cu_seqlens segment over already-rotated q/k/v ([seq, hidden], head-major within a row) — factored
// out so the GLM-OCR tower (glm_ocr_encoder.go), which needs a per-head q/k RMSNorm between the qkv
// split and the rotary, shares the one fused-attention schedule. Moved verbatim.
func attendPackedInto(out, q, k, v []float32, hidden, nH int, cu []int, s *qwenScratch) error {
	hd := hidden / nH
	scale := 1.0 / math.Sqrt(float64(hd))

	attendHead := func(ws *encHeadScratch, mm func(a, b, dst []float32, M, K, N int), head, start, n int) error {
		off := head * hd
		linalg.GatherVBlockMajor(ws.kh, ws.vBlk, k[start*hidden:], v[start*hidden:], head, hd, hidden, n)
		for i := range n {
			gi := start + i
			copy(ws.qh[i*hd:(i+1)*hd], q[gi*hidden+off:gi*hidden+off+hd])
		}
		for i := range n {
			ws.hi[i] = n - 1 // full bidirectional attention within this segment
		}
		if !linalg.AttendTileFusedContractExp(mm, ws.qh[:n*hd], ws.kh[:n*hd], ws.vBlk[:n*hd], ws.ch[:n*hd], ws.fused, n, hd, n, scale, s.loFull[:n], ws.hi[:n]) {
			return fmt.Errorf("vision: AttendTileFused declined for n=%d hd=%d — scratch sizing invariant violated", n, hd)
		}
		for i := range n {
			gi := start + i
			copy(out[gi*hidden+off:gi*hidden+off+hd], ws.ch[i*hd:(i+1)*hd])
		}
		return nil
	}

	workers := len(s.headPool)
	for si := 1; si < len(cu); si++ {
		start, n := cu[si-1], cu[si]-cu[si-1]
		if workers <= 1 {
			ws := &s.headPool[0]
			for head := range nH {
				if err := attendHead(ws, linalg.MatmulBT, head, start, n); err != nil {
					return err
				}
			}
			continue
		}
		var wg sync.WaitGroup
		errs := make([]error, workers)
		headsPer := (nH + workers - 1) / workers
		for w := range workers {
			h0, h1 := w*headsPer, min((w+1)*headsPer, nH)
			if h0 >= h1 {
				continue
			}
			wg.Add(1)
			go func(w, h0, h1 int) {
				defer wg.Done()
				ws := &s.headPool[w]
				for head := h0; head < h1; head++ {
					if err := attendHead(ws, ws.mmWS.MatmulBT, head, start, n); err != nil {
						errs[w] = err
						return
					}
				}
			}(w, h0, h1)
		}
		wg.Wait()
		for _, err := range errs {
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// qwenScratch is the Qwen ViT's per-Forward arena. It mirrors encScratch on the
// SigLIP side, which this encoder was written alongside but never got
// (perf-campaign item 18).
//
// The buffers below were `make([]float32, …)` INSIDE the per-layer loop. make
// zeroes, so that was not merely GC pressure — it was a mandatory memset of
// every byte plus first-touch page faults, per layer. At Qwen2.5-VL ViT dims
// (hidden 1280, inter 3420, 32 layers, ~5184 patches) it comes to ~540 MB per
// layer, ≈15 GB allocated and zeroed per image, of which ~1.5 s is pure memset
// before the collector does anything.
//
// Sized once per Forward. Bit-identical: same operations, same order, distinct
// buffers — the only change is who owns the memory.
type qwenScratch struct {
	n1, n2, o, att []float32        // [np*hidden] block buffers
	mlpOut         []float32        // [np*hidden]
	qkv            []float32        // [np*3*hidden]
	q, k, v        []float32        // [np*hidden]
	headPool       []encHeadScratch // P6: fused-attention worker slots, sized to maxSeg (§attentionInto)
	loFull         []int            // shared, read-only, all-zero: every row's key range starts at 0
	gate, up       []float32        // [np*inter]
}

// newQwenScratch sizes every buffer for the largest shape the forward will use.
// maxSeg is the longest attention segment across BOTH cu_seqlens variants — the
// windowed blocks and the full-attention blocks partition the same patches
// differently, and a layer of either kind may run at any depth.
func newQwenScratch(np, hidden, inter, hd, maxSeg, nH int) *qwenScratch {
	workers := max(min(runtime.GOMAXPROCS(0), nH), 1)
	pool := make([]encHeadScratch, workers)
	for i := range pool {
		pool[i] = newEncHeadScratch(maxSeg, hd)
	}
	return &qwenScratch{
		n1: make([]float32, np*hidden), n2: make([]float32, np*hidden),
		o: make([]float32, np*hidden), att: make([]float32, np*hidden),
		mlpOut: make([]float32, np*hidden),
		qkv:    make([]float32, np*3*hidden),
		q:      make([]float32, np*hidden), k: make([]float32, np*hidden),
		v:        make([]float32, np*hidden),
		headPool: pool, loFull: make([]int, maxSeg),
		gate: make([]float32, np*inter), up: make([]float32, np*inter),
	}
}

// maxSegment returns the longest run in a cu_seqlens boundary list.
func maxSegment(cu []int) int {
	m := 0
	for i := 1; i < len(cu); i++ {
		if l := cu[i] - cu[i-1]; l > m {
			m = l
		}
	}
	return m
}

// mlp runs the gated SiLU MLP: down(silu(gate(x)) * up(x)).
func (e *QwenVisionEncoder) mlpInto(down, x []float32, b *qwenBlock, seq int, s *qwenScratch) {
	hidden, inter := e.Cfg.HiddenSize, e.Cfg.IntermediateSize
	gate := s.gate[:seq*inter]
	b.gatew.MatmulBT(x, gate, seq)
	addBias(gate, b.gateb, seq, inter)
	up := s.up[:seq*inter]
	b.upw.MatmulBT(x, up, seq)
	addBias(up, b.upb, seq, inter)
	n := len(gate)
	parallelChunks(n, func(lo, hi int) {
		g := gate[lo:hi]
		linalg.SiLUContractInto(g, g)
		u := up[lo:hi]
		k := len(g)
		if k > 0 {
			_ = u[k-1]
			i := 0
			for ; i+3 < k; i += 4 {
				g[i+0] *= u[i+0]
				g[i+1] *= u[i+1]
				g[i+2] *= u[i+2]
				g[i+3] *= u[i+3]
			}
			for ; i < k; i++ {
				g[i] *= u[i]
			}
		}
	})
	b.downw.MatmulBT(gate, down, seq)
	addBias(down, b.downb, seq, hidden)
}

func (e *QwenVisionEncoder) isFullAtt(layer int) bool {
	return slices.Contains(e.Cfg.FullattBlockIndexes, layer)
}

// rotaryFreqs builds per-patch rotary frequencies [nPatches][head_dim/2] in
// original patch order: per patch, [h_idx*inv_freq..., w_idx*inv_freq...]. The
// (h_idx,w_idx) coords follow HF get_vision_position_ids — the spatial_merge
// interleave so each merge-unit group's patches are consecutive.
func (e *QwenVisionEncoder) rotaryFreqs(gridTHW [][3]int) []float32 {
	merge := e.Cfg.SpatialMergeSize
	nf := len(e.rotInvFreq) // head_dim/4
	rdim := 2 * nf          // head_dim/2
	var out []float32
	for _, g := range gridTHW {
		t, gridH, gridW := g[0], g[1], g[2]
		block := make([]float32, 0, gridH*gridW*rdim)
		// Spatial-merge interleave: (mbRow,mbCol) index the merge-unit block,
		// (inRow,inCol) the patch within it, so a merge-unit's patches are
		// consecutive (HF get_vision_position_ids order).
		for mbRow := 0; mbRow < gridH/merge; mbRow++ {
			for mbCol := 0; mbCol < gridW/merge; mbCol++ {
				for inRow := range merge {
					for inCol := range merge {
						hpos, wpos := float32(mbRow*merge+inRow), float32(mbCol*merge+inCol)
						for _, f := range e.rotInvFreq {
							block = append(block, hpos*f)
						}
						for _, f := range e.rotInvFreq {
							block = append(block, wpos*f)
						}
					}
				}
			}
		}
		for range t {
			out = append(out, block...)
		}
	}
	return out
}

// windowIndex ports HF get_vision_window_index: groups merge-units into windows of
// (window_size/patch_size/merge)² merged-units, returning the per-group reorder
// indices (length n_patches/merge²) and the cumulative window seqlens (in patch
// units) for the windowed attention blocks.
func (e *QwenVisionEncoder) windowIndex(gridTHW [][3]int) (winIdx, cuWin []int) {
	merge := e.Cfg.SpatialMergeSize
	vmws := e.Cfg.WindowSize / merge / e.Cfg.PatchSize
	mergeUnit := merge * merge
	cuWin = []int{0}
	idOffset := 0
	for _, g := range gridTHW {
		t, gridH, gridW := g[0], g[1], g[2]
		llmH, llmW := gridH/merge, gridW/merge
		// HF pads up to a window multiple; when already divisible it adds a full
		// (all-pad) window that contributes nothing — replicated via count>0 below.
		padH := vmws - llmH%vmws
		padW := vmws - llmW%vmws
		numWinH := (llmH + padH) / vmws
		numWinW := (llmW + padW) / vmws
		for ti := range t {
			for wh := range numWinH {
				for ww := range numWinW {
					count := 0
					for bi := range vmws {
						for bj := range vmws {
							i, j := wh*vmws+bi, ww*vmws+bj
							if i < llmH && j < llmW {
								winIdx = append(winIdx, idOffset+ti*llmH*llmW+i*llmW+j)
								count++
							}
						}
					}
					if count > 0 { // skip all-pad windows (HF unique_consecutive)
						cuWin = append(cuWin, cuWin[len(cuWin)-1]+count*mergeUnit)
					}
				}
			}
		}
		idOffset += t * llmH * llmW
	}
	return winIdx, cuWin
}

// cuSeqlensFull builds the full-attention boundaries: per image, t segments of h*w
// patches each (HF repeat_interleave(h*w, t).cumsum, padded with a leading 0).
func cuSeqlensFull(gridTHW [][3]int) []int {
	cu := []int{0}
	acc := 0
	for _, g := range gridTHW {
		t, gridH, gridW := g[0], g[1], g[2]
		for range t {
			acc += gridH * gridW
			cu = append(cu, acc)
		}
	}
	return cu
}

// --- small f32 helpers specific to the Qwen tower (RMSNorm, SiLU, erf-GELU) ---

// rmsNorm is the weight-only RMSNorm (eps 1e-6) HF uses for the Qwen ViT — variance
// over the last dim, no mean subtraction, no bias. Computed in f64 then cast.
func rmsNorm(x, w []float32, rows, dim int) []float32 {
	out := make([]float32, rows*dim)
	rmsNormInto(out, x, w, rows, dim)
	return out
}

// rmsNormInto is rmsNorm writing into a caller-owned dst[:rows*dim] — the form
// the per-layer loop uses so it does not allocate. dst may not alias x.
// Row-split across cores (audit M-10) — bit-identical for the same reason as
// layerNormInto: the f64 sum-of-squares fold is row-local.
func rmsNormInto(out, x, w []float32, rows, dim int) {
	parallelRows(rows, rows*dim, func(start, end int) {
		rmsNormRows(out, x, w, start, end, dim)
	})
}

func rmsNormRows(out, x, w []float32, start, end, dim int) {
	if dim <= 0 {
		return
	}
	_ = w[dim-1]
	const eps = 1e-6
	dim4 := dim &^ 3
	for r := start; r < end; r++ {
		xr := x[r*dim : r*dim+dim]
		_ = xr[dim-1]
		var ss float64
		d := 0
		for ; d < dim4; d += 4 {
			v0 := float64(xr[d+0])
			v1 := float64(xr[d+1])
			v2 := float64(xr[d+2])
			v3 := float64(xr[d+3])
			ss += v0 * v0
			ss += v1 * v1
			ss += v2 * v2
			ss += v3 * v3
		}
		for ; d < dim; d++ {
			v := float64(xr[d])
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(dim)+eps)
		dst := out[r*dim : r*dim+dim]
		_ = dst[dim-1]
		d = 0
		for ; d < dim4; d += 4 {
			dst[d+0] = float32(float64(xr[d+0])*inv) * w[d+0]
			dst[d+1] = float32(float64(xr[d+1])*inv) * w[d+1]
			dst[d+2] = float32(float64(xr[d+2])*inv) * w[d+2]
			dst[d+3] = float32(float64(xr[d+3])*inv) * w[d+3]
		}
		for ; d < dim; d++ {
			dst[d] = float32(float64(xr[d])*inv) * w[d]
		}
	}
}

// applyRotaryVision applies NeoX rotate_half rotary to one head_dim vector in
// place, given precomputed cos/sin of length head_dim. Body lives in
// rope_scalar.go / rope_simd.go (build-tag dispatched on GOEXPERIMENT=simd);
// this doc comment is the canonical one for both.

// geluErf is the exact (erf) GELU — nn.GELU() default, what the patch merger uses
// (distinct from SigLIP's gelu-tanh).
func geluErf(x []float32) {
	parallelChunks(len(x), func(lo, hi int) {
		linalg.GELUContractInto(x[lo:hi], x[lo:hi])
	})
}

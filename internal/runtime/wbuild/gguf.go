package wbuild

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
)

// loadDeepseekAttnGGUF loads one DeepSeek MLA layer's attention tensors as f32 (the
// parity-first forward uses plain matvec, like the qwen35/granite mixer weights), from
// llama.cpp's deepseek2 names: attn_q_a/attn_q_a_norm/attn_q_b (q-LoRA) or attn_q
// (direct), attn_kv_a_mqa, attn_kv_a_norm, attn_kv_b, attn_output. GGUF stores each matrix
// as dims [in, out] with the data row-major [out, in] — exactly the layout mlaWeights
// expects (and what g.Tensor dequantizes to).
func loadDeepseekAttnGGUF(g *embed.GGUFFile, p string, l *darch.LayerWeights, arch *darch.Architecture) error {
	pa := arch.MLA
	qkHeadDim := pa.QKHeadDim()
	f32 := func(name string, out, in int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return nil, fmt.Errorf("decoder(gguf-deepseek): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return data, nil
	}
	flat := func(name string, n int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-deepseek): %q has %d elems, want %d (dims %v)", name, len(data), n, dims)
		}
		return data, nil
	}
	w := &archparts.MLAWeights{}
	var err error
	if pa.QLoRARank > 0 { // q-LoRA bottleneck (V3 671B); V2-Lite/Moonlight are direct-q
		if w.QAProj, err = f32(p+"attn_q_a.weight", pa.QLoRARank, arch.HiddenDim); err != nil {
			return err
		}
		if w.QALayernorm, err = flat(p+"attn_q_a_norm.weight", pa.QLoRARank); err != nil {
			return err
		}
		if w.QBProj, err = f32(p+"attn_q_b.weight", arch.NumHeads*qkHeadDim, pa.QLoRARank); err != nil {
			return err
		}
	} else {
		if w.QProj, err = f32(p+"attn_q.weight", arch.NumHeads*qkHeadDim, arch.HiddenDim); err != nil {
			return err
		}
	}
	if w.KVAProj, err = f32(p+"attn_kv_a_mqa.weight", pa.KVLoRARank+pa.QKRopeHeadDim, arch.HiddenDim); err != nil {
		return err
	}
	if w.KVALayernorm, err = flat(p+"attn_kv_a_norm.weight", pa.KVLoRARank); err != nil {
		return err
	}
	if w.KVBProj, err = f32(p+"attn_kv_b.weight", arch.NumHeads*(pa.QKNopeHeadDim+pa.VHeadDim), pa.KVLoRARank); err != nil {
		return err
	}
	if w.OProj, err = f32(p+"attn_output.weight", arch.HiddenDim, arch.NumHeads*pa.VHeadDim); err != nil {
		return err
	}
	l.MLA = w
	return nil
}

// ggufRowFiltered reads one row through the dequantizer and applies ggufRowFilter.
func ggufRowFiltered(h *Hooks, name string, into func(start int, dst []float32) error, start int, dst []float32) error {
	if err := into(start, dst); err != nil {
		return err
	}
	if f := h.ggufRowFilter(); f != nil {
		f(name, dst)
	}
	return nil
}

// q4kMat builds the native K-quant WeightMat for QuantQ4K: output rows [0, out) of tensor name, where
// rowSrc(r) is destination row r's element offset in the tensor (a permutation for RoPE-permuted
// q/k, an offset for a fused split). A Q4_K tensor is always kept as Q4_K; a Q5_K or Q6_K one only
// when native56 is set (q4kNative56). ok is false, with a nil error, for any other tensor; the
// caller then loads it at int8 W8A8. The super-blocks are COPIED: loadGGUFWeights closes the
// mapping once the load returns.
func q4kMat(g *embed.GGUFFile, name string, out, in int, rowSrc func(r int) int, native56 bool) (linalg.WeightMat, bool, error) {
	wrap, rowBytes := linalg.WrapQ4K, linalg.Q4KRowBytes
	dims, raw, ok, err := g.Q4KRaw(name)
	if err == nil && !ok && native56 {
		if dims, raw, ok, err = g.Q5KRaw(name); err == nil && ok {
			wrap, rowBytes = linalg.WrapQ5K, linalg.Q5KRowBytes
		} else if err == nil {
			dims, raw, ok, err = g.Q6KRaw(name)
			wrap, rowBytes = linalg.WrapQ6K, linalg.Q6KRowBytes
		}
	}
	if err != nil || !ok {
		return linalg.WeightMat{}, false, err
	}
	if len(dims) != 2 || dims[0] != in || in%256 != 0 {
		return linalg.WeightMat{}, false, fmt.Errorf("decoder(gguf): %q dims %v for a q4k load of [%d, %d]", name, dims, out, in)
	}
	rb := rowBytes(in)
	buf := make([]byte, out*rb)
	for r := range out {
		off := rowSrc(r)
		if off%in != 0 || off/in >= dims[1] {
			return linalg.WeightMat{}, false, fmt.Errorf("decoder(gguf): %q row offset %d outside [%d, %d]", name, off, in, dims[1])
		}
		src := off / in
		copy(buf[r*rb:(r+1)*rb], raw[src*rb:(src+1)*rb])
	}
	wm, err := wrap(buf, out, in)
	return wm, err == nil, err
}

// q4kNative56 reports whether a q4k load keeps the file's Q5_K and Q6_K projections natively instead of widening them to
// int8: only where linalg has an arch kernel for them (linalg.KQuantNative; the portable dot is far slower than int8), and
// only where the reader takes the blocks from the weight itself: a heap load with needCanonical false (no GPU lane; a GPU lane
// on a heap load takes them from the int8 copy through KQuantReader), or a .giw written for the cpu-amd64 or cuda target (kinds
// 11 and 12). A cuda .giw holds no GGUF mapping for KQuantReader, so CUDA's build plans its native weights from the blocks the
// bundle carries (cuda's buildState.planProj); every other target widens.
func q4kNative56(h *Hooks, needCanonical bool, sink *StreamWriter) bool {
	if !linalg.KQuantNative() || h.q4kWiden56() {
		return false
	}
	if sink != nil {
		return giwcodec.CarriesKQuant56(sink.Target)
	}
	return !needCanonical
}

// BuildWeightsFromGGUF dequantizes the GGUF tensors into the weight bundle. When quant is set, each matmul tensor is
// re-quantized (per-row int8 or group-wise int4) right after it is dequantized (and un-permuted) and its f32 is freed, so
// a Q4/Q8 GGUF lands resident as int8/int4 (about 1/4 and 1/8 of f32) without ever materializing the whole model in f32
// (see loadWeights). The GGUF's own quant is lossy and so is the re-quant, but it captures nearly all of what a Q4_K_M
// file carries.
//
// When sink is non-nil, the weights are streamed to it (a .giw body) instead of retained: the header and globals are
// written, then each layer is loaded, written and freed in turn, so peak RAM is about one layer, which is what lets a
// model larger than RAM be transcoded. The returned *Weights then holds no layer tensors; the caller writes the trailing
// CRC. Every family's branch streams. A branch that returns without driving the sink would write a header declaring N
// layers and then none, so any new family branch must stream too; decoder/gguf_streaming_shape_test.go checks its
// closure reads only per-layer tensors, and TestGemma4GGUF_streamedMatchesResident-style byte identity is the gate.
func BuildWeightsFromGGUF(cfg *config.Config, arch *darch.Architecture, g *embed.GGUFFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, sink *StreamWriter, id string, abort <-chan struct{}, h Hooks) (*darch.Weights, error) {
	hidden, hd := arch.HiddenDim, arch.HeadDim
	w := &darch.Weights{Cfg: *cfg, Arch: arch, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	skipRow4 = skipRow4 || !RepackForDecode(sink) // every closure and family branch below takes it from here
	native56 := q4kNative56(&h, needCanonical, sink)

	// streamMat builds a [out, in] linalg.WeightMat by streaming the tensor's rows through
	// the GGUF dequantizer and quantizing each row directly into the resident
	// arrays — no whole-tensor f32 intermediate (see streamQuantized). rowSrc maps
	// a destination row index to its source element offset in the tensor (identity
	// for a plain load; a permutation for the RoPE-permuted q/k projections).
	streamMat := func(name string, out, in int, mode QuantMode, rowSrc func(r int) int) (linalg.WeightMat, error) {
		dims, into, err := GGUFScaledRowDequantizer(g, name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		q4k := mode == QuantQ4K && !g.Has(name+".scale")
		if q4k {
			if wm, ok, err := q4kMat(g, name, out, in, rowSrc, native56); ok || err != nil {
				return wm, err
			}
			mode = QuantInt8I8
		}
		rowInto := func(r int, dst []float32) error { return ggufRowFiltered(&h, name, into, rowSrc(r), dst) }
		// streamMat covers everything mat/permMat do not route to streamMatBatched (o_proj, down_proj, router, any other
		// non-batched, non-embedding layer tensor), so skipRow4 applies here as in quantizeWMSkipRow4.
		wm, err := h.StreamQuantizedLayer(out, in, mode, needCanonical, skipRow4, rowInto)
		if q4k && err == nil && sink == nil {
			w.NoteKQuant(g, name, wm, in, rowSrc)
		}
		return wm, err
	}
	// streamMatBatched is streamMat's twin for attention Q/K/V and MLP gate/up (isBatchedProjTensor):
	// streamQuantizedBatchedProj instead of streamQuantized, so it may build repacked-only int4 when this load's backend
	// allows it. See quantizeBatchedProjWM for the safety argument.
	streamMatBatched := func(name string, out, in int, mode QuantMode, rowSrc func(r int) int) (linalg.WeightMat, error) {
		dims, into, err := GGUFScaledRowDequantizer(g, name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		q4k := mode == QuantQ4K && !g.Has(name+".scale")
		if q4k {
			if wm, ok, err := q4kMat(g, name, out, in, rowSrc, native56); ok || err != nil {
				return wm, err
			}
			mode = QuantInt8I8
		}
		wm, err := h.StreamQuantizedBatchedProj(out, in, mode, needCanonical, skipRow4, func(r int, dst []float32) error {
			return ggufRowFiltered(&h, name, into, rowSrc(r), dst)
		})
		if q4k && err == nil && sink == nil {
			w.NoteKQuant(g, name, wm, in, rowSrc)
		}
		return wm, err
	}
	// mat loads a tensor as a [out, in] linalg.WeightMat, shape-checked, quantized when
	// requested. Routes attn_q/attn_k/attn_v/ffn_gate/ffn_up through streamMatBatched instead
	// of the plain streamMat (isBatchedProjTensor) — see that function's own doc comment.
	mat := func(name string, out, in int) (linalg.WeightMat, error) {
		if IsBatchedProjTensor(name) {
			return streamMatBatched(name, out, in, MatmulQuant(quant, name), func(r int) int { return r * in })
		}
		return streamMat(name, out, in, MatmulQuant(quant, name), func(r int) int { return r * in })
	}
	// embMatAt streams an embedding-shaped table at q through streamQuantizedEmbed, which may build repacked-only when
	// needCanonical is false and int4 mode applies; see that function.
	embMatAt := func(name string, out, in int, q QuantMode) (linalg.WeightMat, error) {
		dims, into, err := GGUFScaledRowDequantizer(g, name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		// into reads the mapping and writes dst alone, so every row range shares it.
		read := func(r int, dst []float32) error { return into(r*in, dst) }
		return h.StreamQuantizedEmbed(out, in, q, needCanonical, !RepackForDecode(sink), 1, StreamEmbedWorkers(sink), func() RowReader { return read })
	}
	// embInto loads an embedding-shaped table into *dst through embMatAt. A streaming build (sink, streamTablesOn) builds nothing:
	// it registers the table's source with the writer, which writes it from the source rows (streamedTable).
	embInto := func(dst *linalg.WeightMat, name string, out, in int, q QuantMode) error {
		if sink == nil || !h.streamTablesOn() {
			m, err := embMatAt(name, out, in, q)
			*dst = m
			return err
		}
		dims, into, err := GGUFScaledRowDequantizer(g, name)
		if err != nil {
			return err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		read := func(r int, dst []float32) error { return into(r*in, dst) }
		if sink.Tables == nil {
			sink.Tables = map[*linalg.WeightMat]*StreamedTable{}
		}
		sink.Tables[dst] = &StreamedTable{rows: out, cols: in, mode: q, newReader: func() RowReader { return read }}
		*dst = linalg.WeightMat{}
		return nil
	}
	// embMat loads the embedding / LM head, which is logit-critical: quantize it with the embedding policy (int8 even in int4
	// mode unless EmbedInt4), not the projection mode.
	embMat := func(dst *linalg.WeightMat, name string, out, in int) error {
		return embInto(dst, name, out, in, quant.EmbeddingWith(embedInt4))
	}
	// vec loads a 1-D tensor (norm or bias).
	vec := func(name string, n int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 1 || dims[0] != n {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [%d]", name, dims, n)
		}
		return data, nil
	}
	// vnorm loads a norm weight. For the Gemma family (RMSAddOne), llama.cpp's GGUF
	// conversion bakes the (1+w) offset into every stored *norm.weight, so this
	// package's RMSAddOne forward would apply it twice — subtract the 1 back out at
	// load to restore the HF convention the shared descriptor expects. No-op for the
	// other architectures (plain RMSNorm), so all norm loads route through here.
	subOneNorm := arch.RMSAddOne
	vnorm := func(name string, n int) ([]float32, error) {
		v, err := vec(name, n)
		if err != nil {
			return nil, err
		}
		if subOneNorm {
			for i := range v {
				v[i] -= 1
			}
		}
		return v, nil
	}
	// permMat loads a q/k projection and inverts llama.cpp's RoPE row permutation.
	// Because the permutation is a pure reorder of whole rows (output features), it
	// commutes with per-row quantization: rather than permute an f32 buffer then
	// quantize, we dequant the rows straight into HF order — destination row hfRow
	// pulls from GGUF source row (h*hd + 2*j + s) — and quantize in place. See
	// ggufInvPermute for the index derivation.
	permMat := func(name string, out, in, nHead int) (linalg.WeightMat, error) {
		// nHead is a validated head count (>0), but a hostile head_dim can still make out/nHead == 1 (half == 0, a divide-by-zero
		// in the row map below) or odd (the RoPE row permutation is only defined for an even head_dim).
		if nHead <= 0 || out%nHead != 0 || (out/nHead)%2 != 0 {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q needs an even head_dim (out=%d, heads=%d)", name, out, nHead)
		}
		hd := out / nHead
		half := hd / 2
		rowSrc := func(hfRow int) int {
			h, rem := hfRow/hd, hfRow%hd
			ggufRow := h*hd + 2*(rem%half) + rem/half
			return ggufRow * in
		}
		if IsBatchedProjTensor(name) { // attn_q/attn_k — see mat's own comment
			return streamMatBatched(name, out, in, MatmulQuant(quant, name), rowSrc)
		}
		return streamMat(name, out, in, MatmulQuant(quant, name), rowSrc)
	}
	// stackedExperts loads a GGUF MoE expert tensor — a 3-D [in, out, nExpert]
	// (fastest-first) blob where each expert occupies a contiguous [out, in]
	// row-major slice — and returns one quantized linalg.WeightMat per expert.
	stackedExperts := func(name string, out, in, nExpert int) ([]linalg.WeightMat, error) {
		dims, into, err := GGUFScaledRowDequantizer(g, name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 3 || dims[0] != in || dims[1] != out || dims[2] != nExpert {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d, experts=%d]", name, dims, in, out, nExpert)
		}
		// Each expert occupies a contiguous [out, in] row-major slice; stream its
		// rows directly into a per-expert quantized linalg.WeightMat (no whole-tensor f32).
		res := make([]linalg.WeightMat, nExpert)
		one := func(e int) error {
			rowInto := func(r int, dst []float32) error { return into((e*out+r)*in, dst) }
			var m linalg.WeightMat
			var err error
			// MoE experts: the same skip-row4-only scope as streamMat.
			m, err = h.StreamQuantizedLayer(out, in, MatmulQuant(quant, name), needCanonical, skipRow4, rowInto)
			if err != nil {
				return err
			}
			res[e] = m
			return nil
		}
		// The dequantizer is a pure function of the mapped tensor bytes and streamQuantized allocates its own scratch, so
		// concurrent experts share nothing and each lands in its own res[e] (forExperts).
		if err := ForExperts(nExpert, sink, one); err != nil {
			return nil, err
		}
		return res, nil
	}
	// fusedSplit streams the output-row range [rowStart, rowStart+rows) of a fused
	// [outTotal, in] tensor into its own quantized linalg.WeightMat — Phi-3's fused
	// attn_qkv (q‖k‖v) and ffn_up (gate‖up), split without a whole-tensor f32 buffer.
	fusedSplit := func(name string, outTotal, in, rowStart, rows int) (linalg.WeightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != outTotal {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf-phi3): %q dims %v, want [in=%d, out=%d]", name, dims, in, outTotal)
		}
		mode := MatmulQuant(quant, name)
		rowSrc := func(r int) int { return (rowStart + r) * in }
		q4k := mode == QuantQ4K
		if q4k {
			if wm, ok, err := q4kMat(g, name, rows, in, rowSrc, native56); ok || err != nil {
				return wm, err
			}
			mode = QuantInt8I8
		}
		rowInto := func(r int, dst []float32) error { return ggufRowFiltered(&h, name, into, rowSrc(r), dst) }
		// Phi-3's fused qkv/gate_up split: the same skip-row4-only scope as streamMat. It is not routed through the
		// isBatchedProjTensor naming convention (GGUF's own fused-tensor special case), so there is no needCanonical or
		// repacked-only branch here.
		var wm linalg.WeightMat
		if skipRow4 {
			wm, err = h.StreamQuantizedSkipRow4(rows, in, mode, rowInto)
		} else {
			wm, err = h.StreamQuantized(rows, in, mode, rowInto)
		}
		if q4k && err == nil && sink == nil {
			w.NoteKQuant(g, name, wm, in, rowSrc)
		}
		return wm, err
	}

	var err error
	if err = embMat(&w.Embed, "token_embd.weight", cfg.VocabSize, hidden); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = vnorm("output_norm.weight", hidden); err != nil {
		return nil, err
	}
	// Separate output head when present; else tied to the embedding.
	arch.TiedLMHead = true
	if g.Has("output.weight") {
		if err = embMat(&w.LMHead, "output.weight", cfg.VocabSize, hidden); err != nil {
			return nil, err
		}
		if quant == QuantQ4K && sink == nil {
			w.NoteKQuant(g, "output.weight", w.LMHead, hidden, func(r int) int { return r * hidden })
		}
		arch.TiedLMHead = false
	}

	// Streaming transcode: emit the header + globals now; each layer is written and
	// freed as it loads (below), so the whole model never sits resident.
	if sink != nil {
		// The writer gates gemma4's head extras and per-layer tail on arch, exactly as writeBundle does
		// for the resident path (it sets wr.Arch = w.arch); a streaming writer built without it would
		// silently drop both.
		sink.Arch = arch
		// gemma4's head carries its model-level PLE inputs, which load in its own branch below — so
		// it writes the head there, once they exist. Every other family's head is complete here.
		if arch.Gemma4 == nil {
			if err := sink.WriteHeadGlobals(w, id); err != nil {
				return nil, err
			}
		}
	}

	b := &ggufBuild{arch: arch, g: g, w: w, quant: quant, needCanonical: needCanonical, skipRow4: skipRow4, sink: sink, abort: abort, h: h,
		hidden: hidden, streamMat: streamMat, mat: mat, vec: vec, vnorm: vnorm, stackedExperts: stackedExperts, hd: hd, permMat: permMat, cfg: cfg, embInto: embInto, id: id, fusedSplit: fusedSplit}

	if q := arch.Qwen35; q != nil {
		return b.loadQwen35(q)
	}

	if arch.GPTOSS != nil {
		return b.loadGptOss()
	}

	if arch.Laguna != nil {
		return b.loadLaguna()
	}

	if gp := arch.Granite; gp != nil {
		return b.loadGranite(gp)
	}

	if npar := arch.Nemotron; npar != nil {
		return b.loadNemotron(npar)
	}

	if g4 := arch.Gemma4; g4 != nil {
		return b.loadGemma4(g4)
	}

	if lp := arch.Llama4; lp != nil {
		return b.loadLlama4(lp)
	}

	return b.loadLayers()
}

// GGUFScaledRowDequantizer wraps g.RowDequantizer(name) and, when a companion `<name>.scale` tensor is present (llama.cpp
// PR #21445's GGUF NVFP4 per-tensor F32 [1] multiplier), scales each dequantized row in place before quantization.
func GGUFScaledRowDequantizer(g *embed.GGUFFile, name string) ([]int, func(offset int, dst []float32) error, error) {
	dims, into, err := g.RowDequantizer(name)
	if err != nil || !g.Has(name+".scale") {
		return dims, into, err
	}
	sDims, sData, sErr := g.Tensor(name + ".scale")
	if sErr != nil {
		return nil, nil, fmt.Errorf("decoder(gguf): %q: %w", name+".scale", sErr)
	}
	if len(sDims) != 1 || sDims[0] != 1 || len(sData) != 1 {
		return nil, nil, fmt.Errorf("decoder(gguf): %q dims %v, want [1]", name+".scale", sDims)
	}
	scale := sData[0]
	if scale == 1 {
		return dims, into, nil
	}
	return dims, func(offset int, dst []float32) error {
		if err := into(offset, dst); err != nil {
			return err
		}
		for i := range dst {
			dst[i] *= scale
		}
		return nil
	}, nil
}

// GGUFQKPermuted reports whether llama.cpp's GGUF conversion permuted this architecture's q/k weights (and their per-row
// biases and head-dim norms). It permutes only for the NORM rope type: llama and dense granite (llama.cpp's GraniteModel
// inherits LlamaModel's undo_permute). The NEOX rope type (qwen2/qwen3/gemma, mellum and the other modern families) leaves
// q/k in HF rotate_half order, so no un-permutation is needed; un-permuting a NEOX file scrambles every head's rotary pairs
// and the output is fluent-looking garbage, not NaN. Unknown archs default to NEOX (no permute), the common modern case.
func GGUFQKPermuted(archName string) bool {
	switch archName {
	case "llama", "granite":
		return true
	default:
		return false
	}
}

// GGUFInvPermuteVec inverts llama.cpp's q/k RoPE permutation for a single
// head_dim-wide vector (a QK-norm weight, shared across heads): HF position
// (s*half + j) comes from GGUF position (2*j + s). Mirrors ggufInvPermute with
// nHead=1 and no input dimension.
func GGUFInvPermuteVec(v []float32) []float32 {
	hd := len(v)
	half := hd / 2
	res := make([]float32, hd)
	for s := range 2 {
		for j := range half {
			res[s*half+j] = v[2*j+s]
		}
	}
	return res
}

// GGUFInvPermute inverts llama.cpp's q/k weight permutation. llama.cpp stores
// the projection rows in interleaved-pair order for its RoPE; this package uses
// HF's rotate_half (first-half / second-half) order. For head h, the HF row
// (s*half + j) — s∈{0,1} selecting the half, j the position within it — comes
// from the GGUF row (2*j + s). w is [out, in] row-major; out = nHead*headDim.
func GGUFInvPermute(w []float32, out, in, nHead int) []float32 {
	hd := out / nHead
	half := hd / 2
	res := make([]float32, len(w))
	for h := range nHead {
		for s := range 2 {
			for j := range half {
				hfRow := h*hd + s*half + j
				ggufRow := h*hd + 2*j + s
				copy(res[hfRow*in:hfRow*in+in], w[ggufRow*in:ggufRow*in+in])
			}
		}
	}
	return res
}

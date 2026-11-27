package vlprep

import (
	"fmt"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// Gemma4MediaRow builds the resident row for one image or audio position of a Gemma 4 E-model: the unscaled projected
// feature followed by PLE inputs computed from PadTokenID.
func Gemma4MediaRow(arch *darch.Architecture, embedLen int, ple func(row []float32, tokenID int, dst []float32), feature []float32) []float32 {
	hidden := arch.HiddenDim
	row := make([]float32, embedLen)
	copy(row[:hidden], feature)
	if len(row) > hidden {
		ple(row[:hidden], arch.Gemma4.PadTokenID, row[hidden:])
	}
	return row
}

// Gemma4EModelImageRows builds the resident rows of a Gemma 4 E-model multimodal prompt. defect is 0 in production and
// non-zero only under planted-defect gate tests (1: placeholder ID for PLE, 2: scaled feature, 3: zero PLE tail, 4: text row).
func Gemma4EModelImageRows(arch *darch.Architecture, embedLen int, ple func(row []float32, tokenID int, dst []float32), embedResident func(int) []float32, ids []int, feats []float32, spans []ImageSpan, defect int) ([][]float32, error) {
	if arch.Gemma4 == nil {
		return nil, fmt.Errorf("decoder: gemma4EModelImageRows on a non-gemma4 model")
	}
	hidden := arch.HiddenDim
	if err := CheckSpans(spans, len(ids), feats, hidden); err != nil {
		return nil, err
	}
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		r, in := SpanRow(spans, i)
		if !in || defect == 4 {
			rows[i] = embedResident(id)
			continue
		}
		f := feats[r*hidden : (r+1)*hidden]
		switch defect {
		case 1:
			row := make([]float32, embedLen)
			copy(row[:hidden], f)
			ple(row[:hidden], id, row[hidden:])
			rows[i] = row
		case 2:
			sc := float32(1)
			if e := arch.EmbedScale; e != 0 && e != 1 {
				sc = float32(e)
			}
			g := make([]float32, hidden)
			for k := range g {
				g[k] = f[k] * sc
			}
			rows[i] = Gemma4MediaRow(arch, embedLen, ple, g)
		case 3:
			row := Gemma4MediaRow(arch, embedLen, ple, f)
			clear(row[hidden:])
			rows[i] = row
		default:
			rows[i] = Gemma4MediaRow(arch, embedLen, ple, f)
		}
	}
	return rows, nil
}

// Gemma4BidirImageRows builds the resident rows of a bidirectional (non-E) Gemma 4 multimodal prompt. defect is 0 in
// production and 2 when a planted-defect test scales image features by embedScale.
func Gemma4BidirImageRows(arch *darch.Architecture, embedScale float64, embedN func([]int) []float32, ids []int, feats []float32, spans []ImageSpan, defect int) ([][]float32, error) {
	if arch.Gemma4 == nil {
		return nil, fmt.Errorf("decoder: gemma4BidirImageRows on a non-gemma4 model")
	}
	hidden := arch.HiddenDim
	if err := CheckSpans(spans, len(ids), feats, hidden); err != nil {
		return nil, err
	}
	h := embedN(ids)
	Splice(h, spans, feats, hidden)
	if defect == 2 && embedScale != 0 && embedScale != 1 {
		for _, s := range spans {
			for i := s.Pos * hidden; i < (s.Pos+s.Len)*hidden; i++ {
				h[i] *= float32(embedScale)
			}
		}
	}
	rows := make([][]float32, len(ids))
	for i := range rows {
		rows[i] = h[i*hidden : (i+1)*hidden : (i+1)*hidden]
	}
	return rows, nil
}

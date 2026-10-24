//go:build gpu

package gpu

import "strings"

// quantNarrowForTest makes New's Context compile the single-workgroup quantizers at 64 lanes, as their source is written (the
// old arm of KS-E2's A/B); read at New.
var quantNarrowForTest bool

// quantSrc is a single-workgroup quantizer's source as this Context compiles it: widened to 256 lanes (widenQuant), or the
// 64-lane source on a quantNarrow Context.
func (c *Context) quantSrc(label, src string) (string, error) {
	if c.quantNarrow {
		return src, nil
	}
	return widenQuant(label, src)
}

// widenQuant runs a 64-lane single-workgroup quantizer (rmsnormQuantWGSL, swigluQuantWGSL, gegluQuantWGSL, quantizeShaderWGSL,
// relu2QuantWGSL, gptossGluQuantWGSL) on 256 lanes: the shared array, the workgroup size, the max loop and the pack loop at
// 256, the max tree from stride 128. f32 max is exact in any order and each packed value depends on its own element, so the
// output is the 64-lane kernel's bits. rmsnormQuant's sum of squares is order-dependent: it stays on lanes 0-63 with its
// 64-lane stride and tree. The edits are exact-count, so a source that changed shape is refused rather than half-widened.
func widenQuant(label, src string) (string, error) {
	edits := []srcEdit{
		{"var<workgroup> sh: array<f32, 64>;", "var<workgroup> sh: array<f32, 256>;", 1},
		{"@compute @workgroup_size(64)", "@compute @workgroup_size(256)", 1},
	}
	if strings.Contains(src, "ss = ss + v*v;") {
		edits = append(edits,
			srcEdit{"for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[i]; ss = ss + v*v; }",
				"if (t < 64u) { for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[i]; ss = ss + v*v; } }", 1},
			srcEdit{"for (var i: u32 = t; i < p.h; i = i + 64u) {\n        var w = weight[i];",
				"for (var i: u32 = t; i < p.h; i = i + 256u) {\n        var w = weight[i];", 1},
			srcEdit{"    stride = 32u;\n", "    stride = 128u;\n", 1},
		)
	} else {
		edits = append(edits,
			srcEdit{"i = i + 64u)", "i = i + 256u)", 1},
			srcEdit{"var stride: u32 = 32u;", "var stride: u32 = 128u;", 1},
		)
	}
	if strings.Contains(src, "w = w + 64u)") && !strings.Contains(src, "wi = wi + 64u)") {
		edits = append(edits, srcEdit{"w = w + 64u)", "w = w + 256u)", 1})
	} else {
		edits = append(edits, srcEdit{"wi = wi + 64u)", "wi = wi + 256u)", 1})
	}
	return editSource(label, src, edits)
}

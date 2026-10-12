//go:build gpu

package gpu

import (
	"fmt"
	"strings"
)

// Decode quantizes an activation inside the kernel that computes it (rmsnormQuantWGSL, swigluQuantWGSL,
// gegluQuantWGSL): the float value is recomputed in the packing loop and never stored. A pass that stores the value and
// quantizes it in a second kernel rounds a rare value to the neighbouring int8 step, and from that row on its output
// is no longer decode's. The batched pass therefore runs these kernels' own source, one row a workgroup.

// srcEdit is one textual edit of a kernel's source: old must occur exactly n times.
type srcEdit struct {
	old, new string
	n        int
}

// editSource applies edits to src, refusing when the source no longer has the text an edit names.
func editSource(label, src string, edits []srcEdit) (string, error) {
	for _, e := range edits {
		if got := strings.Count(src, e.old); got != e.n {
			return "", fmt.Errorf("gpu: %s: the single-row kernel's source has %q %d times, not %d; its rows form must be re-derived", label, e.old, got, e.n)
		}
		src = strings.ReplaceAll(src, e.old, e.new)
	}
	return src, nil
}

const fusedRowsMain = "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n    let t = lid.x;\n    let rb = wid.x * "

// fusedRowsKind names a fused quantizing kernel in its one-row-a-workgroup form.
type fusedRowsKind int

const (
	fusedRowsRMS    fusedRowsKind = iota // rmsnormQuantWGSL: src [M, h], weight [h], qout [M, np/4], scales [M]
	fusedRowsSwiGLU                      // swigluQuantWGSL: gate [M, n], up [M, n], qout [M, np/4], scales [M]
	fusedRowsGeGLU                       // gegluQuantWGSL, the same bindings
	fusedRowsKinds
)

// fusedRowsSource derives kind's shader from decode's single-row kernel: every row-indexed read and write gains the
// row's offset, and nothing else changes. Dispatched with one workgroup a row, grid (M, 1).
func fusedRowsSource(kind fusedRowsKind) (code, label string, err error) {
	const oldMain = "fn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n    let t = lid.x;"
	out := []srcEdit{
		{"scales[0] = s;", "scales[wid.x] = s;", 1},
		{"qout[wi] = word;", "qout[wid.x * nw + wi] = word;", 1},
	}
	if kind == fusedRowsRMS {
		label = "rmsnormQuantRows"
		code, err = editSource(label, rmsnormQuantWGSL, append([]srcEdit{
			{oldMain, fusedRowsMain + "p.h;", 1},
			{"let v = src[i];", "let v = src[rb + i];", 1},
			{"abs(src[i] * inv * w)", "abs(src[rb + i] * inv * w)", 1},
			{"round(src[k] * inv * w * invs)", "round(src[rb + k] * inv * w * invs)", 1},
		}, out...))
		return code, label, err
	}
	src, label := swigluQuantWGSL, "swigluQuantRows"
	if kind == fusedRowsGeGLU {
		src, label = gegluQuantWGSL, "gegluQuantRows"
	}
	code, err = editSource(label, src, append([]srcEdit{
		{oldMain, fusedRowsMain + "p.n;", 1},
		{"(gate[i], up[i])", "(gate[rb + i], up[rb + i])", 1},
		{"(gate[k], up[k])", "(gate[rb + k], up[rb + k])", 1},
	}, out...))
	return code, label, err
}

// ensureFusedRows compiles the rows forms of decode's fused quantizing kernels.
func (c *Context) ensureFusedRows() error {
	for kind := range c.fusedRows {
		if c.fusedRows[kind].pl != nil {
			continue
		}
		code, label, err := fusedRowsSource(fusedRowsKind(kind))
		if err != nil {
			return err
		}
		sh, pl, layout, err := c.mkPipeline(label, code)
		if err != nil {
			return err
		}
		c.fusedRows[kind] = tiledPipes{sh: sh, pl: pl, layout: layout}
	}
	return nil
}

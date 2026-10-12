//go:build gpu

package gpu

import (
	"fmt"
	"strings"
)

// attnKeysRowsSource is the rows form of a key-split attention kernel, the narrow one or the wide one: every row of a
// packed [M, nH*hd] query buffer in one dispatch, grid (nH, M). A workgroup's row is its y coordinate; the row's key
// count is the first row's plus the row, and on a sliding-window layer (p.window not 0) its first key is the row's own,
// as decode computes it a token. Nothing else is edited, so a row's context is the single-row kernel's.
func attnKeysRowsSource(wide bool) (string, error) {
	src, label := attnKeysShaderWGSL, "attnKeysRows"
	ctxOld, ctxNew := "ctx[qh * hd + t]", "ctx[row * p.nH * hd + qh * hd + t]"
	if wide {
		var err error
		if src, err = attnKeysWideSource(); err != nil {
			return "", err
		}
		label = "attnKeysWideRows"
		ctxOld, ctxNew = "ctx[qh * hd + d]", "ctx[row * p.nH * hd + qh * hd + d]"
	}
	out, err := editSource(label, src, []srcEdit{
		{"scale: f32, _p: u32 };", "scale: f32, window: u32 };", 1},
		{"if (tileStart >= p.nKeys) { break; }", "if (tileStart >= rowKeys) { break; }", 1},
		{"if (tileEnd > p.nKeys) { tileEnd = p.nKeys; }", "if (tileEnd > rowKeys) { tileEnd = rowKeys; }", 1},
		{"var tileStart: u32 = p.start;", "var tileStart: u32 = rowStart;", 1},
		{"let qb4 = (qh * hd) / 4u;", "let qb4 = (row * p.nH * hd + qh * hd) / 4u;", 1},
		{ctxOld, ctxNew, 1},
		{"    let qh = wid.x;\n", "    let qh = wid.x;\n" +
			"    let row = wid.y;\n" +
			"    let rowKeys = p.nKeys + row;   // the uniform carries the first row's key count\n" +
			"    var rowStart = p.start;\n" +
			"    if (p.window != 0u && rowKeys > p.window && rowKeys - p.window > rowStart) { rowStart = rowKeys - p.window; }\n", 1},
	})
	if err != nil {
		return "", err
	}
	// Every other read of the single row's key count or start would be a row reading the first row's.
	if strings.Count(out, "p.nKeys") != 1 || strings.Count(out, "p.start") != 1 {
		return "", fmt.Errorf("gpu: %s: the single-row kernel reads p.nKeys or p.start somewhere its rows form does not edit", label)
	}
	return out, nil
}

// ensureAttnKeysRows compiles the rows forms of the two key-split attention kernels.
func (c *Context) ensureAttnKeysRows() error {
	for i, wide := range []bool{false, true} {
		if c.attnKeysRows[i].pl != nil {
			continue
		}
		src, err := attnKeysRowsSource(wide)
		if err != nil {
			return err
		}
		sh, pl, ly, err := c.mkPipeline(map[bool]string{false: "attn-keys-rows", true: "attn-keys-wide-rows"}[wide], src)
		if err != nil {
			return err
		}
		c.attnKeysRows[i] = tiledPipes{sh: sh, pl: pl, layout: ly}
	}
	return nil
}

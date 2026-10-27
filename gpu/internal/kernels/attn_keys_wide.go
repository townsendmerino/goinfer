//go:build gpu

package kernels

// attnKeysWideSource is attnKeysShaderWGSL for a head of 129 to 256 dimensions. The key-split kernel's scores and
// softmax do not depend on the head's width; only its last step does, where each of the 128 lanes owns one output
// dimension. Here a lane owns two, t and t+128, each summed over the keys exactly as the narrow kernel sums its one.
// Everything else is the narrow kernel's own text.
func attnKeysWideSource() (string, error) {
	return editSource("attnKeysWide", attnKeysShaderWGSL, []srcEdit{
		{`    var acc: f32 = 0.0;   // lane t's running weighted V-sum for dim t (t < hd)`,
			`    var acc: array<f32, 2>;   // lane t's running weighted V-sums for dims t and t+128
    acc[0] = 0.0;
    acc[1] = 0.0;`, 1},
		{`        if (t < hd) {
            var a: f32 = acc * corr;
            for (var s: u32 = tileStart; s < tileEnd; s = s + 1u) {
                a = a + sc[s - tileStart] * vals[s * kvDim + kvbase + t];
            }
            acc = a;
        }`,
			`        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let d = t + j * 128u;
            if (d < hd) {
                var a: f32 = acc[j] * corr;
                for (var s: u32 = tileStart; s < tileEnd; s = s + 1u) {
                    a = a + sc[s - tileStart] * vals[s * kvDim + kvbase + d];
                }
                acc[j] = a;
            }
        }`, 1},
		{`    if (t < hd) { ctx[qh * hd + t] = acc / l; }`,
			`    for (var j: u32 = 0u; j < 2u; j = j + 1u) {
        let d = t + j * 128u;
        if (d < hd) { ctx[qh * hd + d] = acc[j] / l; }
    }`, 1},
	})
}

// attnKeysWideEligible reports whether the wide key-split kernel can serve this geometry: a float32 K/V cache, a head
// wider than the narrow kernel's and no wider than two dimensions a lane, and hd and kvDim multiples of 4 for the vec4
// loads (attnKeysEligible's own condition).
func attnKeysWideEligible(hd, kvDim int, kvF16 bool) bool {
	return !kvF16 && hd > AttnWG && hd <= 2*AttnWG && hd%4 == 0 && kvDim%4 == 0
}

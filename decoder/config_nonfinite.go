package decoder

import "bytes"

// lenientConfigJSON rewrites the bare tokens Infinity, -Infinity and NaN that Python's json.dump writes into a model's config.json
// (Nemotron-H's time_step_limit is [0.0, Infinity]) into JSON a Go decoder accepts: the largest finite float64, its negation, and null.
// Only tokens outside strings are touched, and only whole tokens. It is for model config files; request bodies are decoded strictly and
// never pass through it. A config without those tokens is returned as it came.
func lenientConfigJSON(b []byte) []byte {
	if !bytes.Contains(b, []byte("Infinity")) && !bytes.Contains(b, []byte("NaN")) {
		return b
	}
	const maxFloat = "1.7976931348623157e308"
	out := make([]byte, 0, len(b)+16)
	inStr := false
	for i := 0; i < len(b); {
		c := b[i]
		if inStr {
			out = append(out, c)
			switch {
			case c == '\\' && i+1 < len(b):
				out = append(out, b[i+1])
				i += 2
				continue
			case c == '"':
				inStr = false
			}
			i++
			continue
		}
		if c == '"' {
			inStr = true
			out = append(out, c)
			i++
			continue
		}
		if tok, rep := nonFiniteAt(b, i, maxFloat); tok > 0 {
			out = append(out, rep...)
			i += tok
			continue
		}
		out = append(out, c)
		i++
	}
	return out
}

// nonFiniteAt reports the length of the non-finite token starting at b[i] and its replacement, or 0. A token is a whole word: the byte before
// it is not part of an identifier or number, and the byte after it is not part of an identifier.
func nonFiniteAt(b []byte, i int, maxFloat string) (int, string) {
	if i > 0 && isWordByte(b[i-1]) {
		return 0, ""
	}
	for _, t := range []struct{ text, rep string }{{"-Infinity", "-" + maxFloat}, {"Infinity", maxFloat}, {"NaN", "null"}} {
		n := len(t.text)
		if i+n <= len(b) && string(b[i:i+n]) == t.text && (i+n == len(b) || !isWordByte(b[i+n])) {
			return n, t.rep
		}
	}
	return 0, ""
}

func isWordByte(c byte) bool {
	return c == '_' || c == '.' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

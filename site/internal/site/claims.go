package site

import (
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Section returns the text under the heading in a markdown file: from the heading line to the next heading of the same
// or a higher level. The heading matches on its text, exactly, after the leading #s.
func Section(md, heading string) (string, bool) {
	lines := strings.Split(md, "\n")
	level := -1
	start := -1
	for i, ln := range lines {
		l, txt, ok := headingOf(ln)
		if !ok {
			continue
		}
		if level < 0 {
			if txt == heading {
				level, start = l, i+1
			}
			continue
		}
		if l <= level {
			return strings.Join(lines[start:i], "\n"), true
		}
	}
	if level < 0 {
		return "", false
	}
	return strings.Join(lines[start:], "\n"), true
}

func headingOf(ln string) (level int, text string, ok bool) {
	n := 0
	for n < len(ln) && ln[n] == '#' {
		n++
	}
	if n == 0 || n > 6 || n >= len(ln) || ln[n] != ' ' {
		return 0, "", false
	}
	return n, strings.TrimSpace(ln[n+1:]), true
}

// hasToken reports whether tok appears in text as a whole number: not inside a longer one ("253.1" is not in "1253.1"
// or "253.12"). A sentence's trailing full stop is not part of the number.
func hasToken(text, tok string) bool {
	re := regexp.MustCompile(`(?:^|[^0-9.])` + regexp.QuoteMeta(tok) + `(?:$|[^0-9.]|\.(?:$|[^0-9]))`)
	return re.MatchString(text)
}

// round2 renders a decimal string to 2 places, half away from zero, exactly (no float error: 1.815 is 1.82).
func round2(s string) (string, error) {
	r, ok := new(big.Rat).SetString(s)
	if !ok {
		return "", fmt.Errorf("%q is not a number", s)
	}
	r.Mul(r, big.NewRat(100, 1))
	half := big.NewRat(1, 2)
	r.Add(r, half)
	f := new(big.Float).SetRat(r)
	i, _ := f.Int(nil) // truncates: the value is positive, so this floors
	q := new(big.Rat).SetFrac(i, big.NewInt(100))
	return q.FloatString(2), nil
}

// medianOf reads "a / b / c" and returns the median of the three.
func medianOf(runs string) (float64, error) {
	parts := strings.Split(runs, "/")
	if len(parts) != 3 {
		return 0, fmt.Errorf("%q is not three runs", runs)
	}
	v := make([]float64, 3)
	for i, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return 0, err
		}
		v[i] = f
	}
	sort.Float64s(v)
	return v[1], nil
}

func fnum(p *float64) string { return strconv.FormatFloat(*p, 'f', -1, 64) }

// CheckClaims is the claims check (S8b): every figure the site shows must be in the repo file it cites, under the
// heading it cites, and the claim's date must be there too. It is what stops a number from drifting away from its
// source, or from being typed in from memory: the site's own speeds failed it the first time it ran.
func CheckClaims(root string, in *Inputs) error {
	cache := map[string]string{}
	section := func(s Source) (string, error) {
		md, ok := cache[s.Path]
		if !ok {
			b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(s.Path)))
			if err != nil {
				return "", fmt.Errorf("cited file: %w", err)
			}
			md = string(b)
			cache[s.Path] = md
		}
		sec, found := Section(md, s.Heading)
		if !found {
			return "", fmt.Errorf("heading %q is not in %s", s.Heading, s.Path)
		}
		return sec, nil
	}
	var bad []string
	for _, c := range in.Claims.Claims {
		sec, err := section(c.Source)
		if err != nil {
			bad = append(bad, fmt.Sprintf("claim %s: %v", c.ID, err))
			continue
		}
		need := map[string]string{"date": c.Date}
		if c.Value != nil {
			need["value"] = fnum(c.Value)
		}
		if c.Peer != nil {
			need["peer"] = fnum(c.Peer)
		}
		if c.RatioRaw != "" {
			need["ratio_raw"] = c.RatioRaw
		}
		for what, tok := range need {
			ok := hasToken(sec, tok)
			if what == "date" {
				ok = strings.Contains(sec, tok)
			}
			if !ok {
				bad = append(bad, fmt.Sprintf("claim %s: %s %q is not under %q in %s", c.ID, what, tok, c.Source.Heading, c.Source.Path))
			}
		}
		if c.Value != nil && c.RatioRaw != "" {
			want, err := round2(c.RatioRaw)
			if err == nil && (c.Verdict == "ahead" || c.Verdict == "behind") && c.Ratio != want+"×" {
				bad = append(bad, fmt.Sprintf("claim %s: shows %q but its raw ratio %s rounds to %s×", c.ID, c.Ratio, c.RatioRaw, want))
			}
		}
		switch c.Basis {
		case "median-of-3":
			for _, pair := range []struct {
				name string
				runs string
				val  *float64
			}{{"value", c.Runs, c.Value}, {"peer", c.PeerRuns, c.Peer}} {
				if pair.val == nil {
					continue
				}
				if !strings.Contains(sec, pair.runs) {
					bad = append(bad, fmt.Sprintf("claim %s: the %s runs %q are not under %q in %s", c.ID, pair.name, pair.runs, c.Source.Heading, c.Source.Path))
					continue
				}
				med, err := medianOf(pair.runs)
				if err != nil {
					bad = append(bad, fmt.Sprintf("claim %s: %s runs: %v", c.ID, pair.name, err))
				} else if med != *pair.val {
					bad = append(bad, fmt.Sprintf("claim %s: the %s is %s but the median of its runs (%s) is %s", c.ID, pair.name, fnum(pair.val), pair.runs, strconv.FormatFloat(med, 'f', -1, 64)))
				}
			}
		case "mean", "none":
		default:
			bad = append(bad, fmt.Sprintf("claim %s: basis %q must be median-of-3, mean or none", c.ID, c.Basis))
		}
		if (c.Verdict == "ahead" || c.Verdict == "behind") && (c.Value == nil || c.Peer == nil) {
			bad = append(bad, fmt.Sprintf("claim %s: a verdict against Ollama needs both figures", c.ID))
		}
	}
	for _, f := range in.Claims.Facts {
		sec, err := section(f.Source)
		if err != nil {
			bad = append(bad, fmt.Sprintf("fact %q: %v", f.Text, err))
			continue
		}
		if !strings.Contains(sec, f.Text) {
			bad = append(bad, fmt.Sprintf("fact %q is not under %q in %s", f.Text, f.Source.Heading, f.Source.Path))
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("the claims check failed (%d):\n  %s", len(bad), strings.Join(bad, "\n  "))
	}
	return nil
}

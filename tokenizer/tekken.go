package tokenizer

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

// Mistral's Tekken tokenizer (Voxtral Mini, Mistral Small 3.x), shipped as ONE file, tekken.json, and no tokenizer.json (docs/tasks/task-multimodal-support-2026-10.md).
//
// The file is tiktoken's format: a list of byte strings whose position is their RANK, plus 1,000 special tokens at ids 0-999. The model's token id is rank+1000, and only the first
// default_vocab_size-1000 ranks are in the model's vocabulary (130,072 of the 150,000 shipped). tiktoken encodes by repeatedly merging the adjacent pair whose concatenation has the
// LOWEST rank; this package's core merges by an ordered merge list. The two are the same machine when the list is derived from the ranks the way transformers' TikTokenConverter does
// (every multi-byte token is the merge of the two parts its own rank splits into when merging stops one step short of it), sorted by the merged token's rank, with `ignore_merges` so a
// piece that is a whole token wins outright, as it does in tiktoken. This builds that document in memory and hands it to the ordinary parser, as LoadVocabMerges does for Qwen3-ASR.
//
// The special tokens are NOT parsed out of plain text by mistral_common (a user's "[INST]" is text), so callers encode content through EncodeSegments with a non-special segment and build
// the structural ids themselves (multimodal.VoxtralPrompt); Encode would parse them.

// ErrNotTekken is returned when a file is not a Tekken tokenizer this package knows how to assemble.
var ErrNotTekken = errors.New("tokenizer: not a tekken.json this build understands")

type tekkenFile struct {
	Config struct {
		Pattern           string `json:"pattern"`
		NumVocabTokens    int    `json:"num_vocab_tokens"`
		DefaultVocabSize  int    `json:"default_vocab_size"`
		DefaultNumSpecial int    `json:"default_num_special_tokens"`
		Version           string `json:"version"`
	} `json:"config"`
	Vocab []struct {
		Rank       int    `json:"rank"`
		TokenBytes string `json:"token_bytes"`
	} `json:"vocab"`
	Special []struct {
		Rank      int    `json:"rank"`
		TokenStr  string `json:"token_str"`
		IsControl bool   `json:"is_control"`
	} `json:"special_tokens"`
}

// tekkenOptions are the build's switches. The zero value is the correct build; the planted-defect tests flip one switch at a time (tekken_test.go) so a gate that cannot see the mistake is
// found before a real model depends on it.
type tekkenOptions struct {
	rankOffsetOverride *int // nil: the file's special-token count
	noTruncate         bool // keep all 150,000 ranks instead of default_vocab_size - specials
	sortBySize         bool // order the merges by the merged token's length, not its rank
}

// LoadTekken reads a Tekken tokenizer from the bytes of a tekken.json.
func LoadTekken(raw []byte) (*Tokenizer, error) { return loadTekken(raw, tekkenOptions{}) }

func loadTekken(raw []byte, opt tekkenOptions) (*Tokenizer, error) {
	var f tekkenFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNotTekken, err)
	}
	if f.Config.Pattern == "" || len(f.Vocab) == 0 || len(f.Special) == 0 {
		return nil, fmt.Errorf("%w: missing pattern, vocab or special_tokens", ErrNotTekken)
	}
	nSpecial := f.Config.DefaultNumSpecial
	if nSpecial <= 0 || nSpecial != len(f.Special) {
		return nil, fmt.Errorf("%w: default_num_special_tokens %d with %d special tokens", ErrNotTekken, nSpecial, len(f.Special))
	}
	offset := nSpecial
	if opt.rankOffsetOverride != nil {
		offset = *opt.rankOffsetOverride
	}
	nRanks := len(f.Vocab)
	if !opt.noTruncate && f.Config.DefaultVocabSize > nSpecial && f.Config.DefaultVocabSize-nSpecial < nRanks {
		nRanks = f.Config.DefaultVocabSize - nSpecial
	}
	toks := make([][]byte, nRanks)
	ranks := make(map[string]int, nRanks) // token bytes -> rank
	for i := 0; i < nRanks; i++ {
		e := f.Vocab[i]
		if e.Rank != i {
			return nil, fmt.Errorf("%w: vocab entry %d carries rank %d", ErrNotTekken, i, e.Rank)
		}
		b, err := base64.StdEncoding.DecodeString(e.TokenBytes)
		if err != nil || len(b) == 0 {
			return nil, fmt.Errorf("%w: vocab entry %d token_bytes: %v", ErrNotTekken, i, err)
		}
		toks[i] = b
		ranks[string(b)] = i
	}
	enc, _ := buildByteLevelTables() // byte -> printable rune
	mapped := func(b []byte) string {
		rs := make([]rune, len(b))
		for i, c := range b {
			rs[i] = enc[c]
		}
		return string(rs)
	}
	vocab := make(map[string]int32, nRanks)
	for i, b := range toks {
		vocab[mapped(b)] = int32(i + offset)
	}

	// The merge list: for each multi-byte token, merge its bytes by lowest rank but stop before the token's own rank; two parts must remain.
	type merge struct {
		rank int
		a, b string
		size int
	}
	merges := make([]merge, 0, nRanks)
	for r := 256; r < nRanks; r++ { // ranks 0-255 are the single bytes
		tok := toks[r]
		parts := make([][]byte, len(tok))
		for i := range tok {
			parts[i] = tok[i : i+1]
		}
		for len(parts) > 2 {
			best, bestRank := -1, -1
			for i := 0; i+1 < len(parts); i++ {
				cat := append(append(make([]byte, 0, len(parts[i])+len(parts[i+1])), parts[i]...), parts[i+1]...)
				if rk, ok := ranks[string(cat)]; ok && rk < r && (bestRank < 0 || rk < bestRank) {
					best, bestRank = i, rk
				}
			}
			if best < 0 {
				break
			}
			cat := append(append([]byte(nil), parts[best]...), parts[best+1]...)
			parts = append(append(parts[:best:best], cat), parts[best+2:]...)
		}
		if len(parts) != 2 {
			continue // not derivable from lower ranks: reachable only as a whole piece (ignore_merges)
		}
		merges = append(merges, merge{rank: r, a: mapped(parts[0]), b: mapped(parts[1]), size: len(tok)})
	}
	if opt.sortBySize {
		sort.SliceStable(merges, func(i, j int) bool { return merges[i].size > merges[j].size })
	} else {
		sort.Slice(merges, func(i, j int) bool { return merges[i].rank < merges[j].rank })
	}
	pairs := make([][2]string, len(merges))
	for i, m := range merges {
		pairs[i] = [2]string{m.a, m.b}
	}

	type added struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	}
	at := make([]added, 0, len(f.Special))
	for _, s := range f.Special {
		at = append(at, added{ID: int32(s.Rank), Content: s.TokenStr, Special: true})
	}
	doc := map[string]any{
		"version":      "1.0",
		"added_tokens": at,
		"pre_tokenizer": map[string]any{"type": "Sequence", "pretokenizers": []any{
			map[string]any{"type": "Split", "pattern": map[string]any{"Regex": f.Config.Pattern}, "behavior": "Isolated", "invert": false},
			map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
		}},
		"decoder": map[string]any{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
		"model":   map[string]any{"type": "BPE", "vocab": vocab, "merges": pairs, "byte_fallback": false, "ignore_merges": true},
	}
	out, err := json.Marshal(doc)
	if err != nil {
		return nil, err
	}
	t, err := parseTokenizerJSON(out, "tekken.json", "")
	if err != nil {
		return nil, err
	}
	t.special.BOS, t.special.EOS = -1, -1
	for _, s := range f.Special {
		switch s.TokenStr {
		case "<s>":
			t.special.BOS = s.Rank
		case "</s>":
			t.special.EOS = s.Rank
		}
	}
	return t, nil
}

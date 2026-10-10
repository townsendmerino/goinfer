package tokenizer

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// A Qwen2-style tokenizer shipped as vocab.json + merges.txt + tokenizer_config.json, with no tokenizer.json (Qwen3-ASR's repo is one;
// docs/tasks/task-multimodal-support-2026-10.md). transformers builds the fast tokenizer from those three files with a fixed pipeline for tokenizer_class Qwen2Tokenizer: NFC
// normalisation, the Qwen2 split regex, byte-level BPE, a byte-level decoder, the config's added tokens. This composes the same tokenizer.json in memory and hands it to the
// ordinary parser, so one code path reads both. It is deliberately limited to that class: the split regex is part of the tokenizer's identity (GPT-2, Llama 3 and Qwen2 differ),
// and guessing it for an unknown class would be a silent wrong tokenization.

// qwen2SplitRegex is the pre-tokenizer pattern of tokenizer_class Qwen2Tokenizer (Qwen2, Qwen2.5, Qwen3, Qwen3-ASR), as transformers writes it into the tokenizer.json it builds.
const qwen2SplitRegex = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// ErrNoVocabMerges is returned when a directory is not a vocab.json + merges.txt tokenizer this package knows how to assemble.
var ErrNoVocabMerges = errors.New("tokenizer: no tokenizer.json, and no vocab.json + merges.txt of a supported class")

// LoadVocabMerges assembles the tokenizer of a directory that has vocab.json, merges.txt and a tokenizer_config.json whose tokenizer_class is Qwen2Tokenizer or Qwen2TokenizerFast.
func LoadVocabMerges(dir string) (*Tokenizer, error) {
	cfgRaw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNoVocabMerges, err)
	}
	var cfg struct {
		Class string `json:"tokenizer_class"`
		Added map[string]struct {
			Content    string `json:"content"`
			LStrip     bool   `json:"lstrip"`
			RStrip     bool   `json:"rstrip"`
			Normalized bool   `json:"normalized"`
			SingleWord bool   `json:"single_word"`
			Special    bool   `json:"special"`
		} `json:"added_tokens_decoder"`
	}
	if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
		return nil, fmt.Errorf("tokenizer: parse %s/tokenizer_config.json: %w", dir, err)
	}
	if cfg.Class != "Qwen2Tokenizer" && cfg.Class != "Qwen2TokenizerFast" {
		return nil, fmt.Errorf("%w (tokenizer_class %q)", ErrNoVocabMerges, cfg.Class)
	}
	vocabRaw, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNoVocabMerges, err)
	}
	vocab := map[string]int32{}
	if err := json.Unmarshal(vocabRaw, &vocab); err != nil {
		return nil, fmt.Errorf("tokenizer: parse %s/vocab.json: %w", dir, err)
	}
	mf, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNoVocabMerges, err)
	}
	defer mf.Close()
	var merges [][2]string
	sc := bufio.NewScanner(mf)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r")
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		a, b, ok := strings.Cut(line, " ")
		if !ok || a == "" || b == "" || strings.Contains(b, " ") {
			return nil, fmt.Errorf("tokenizer: %s/merges.txt: malformed merge %q", dir, line)
		}
		merges = append(merges, [2]string{a, b})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	type added struct {
		ID         int32  `json:"id"`
		Content    string `json:"content"`
		SingleWord bool   `json:"single_word"`
		LStrip     bool   `json:"lstrip"`
		RStrip     bool   `json:"rstrip"`
		Normalized bool   `json:"normalized"`
		Special    bool   `json:"special"`
	}
	var at []added
	for idStr, a := range cfg.Added {
		var id int32
		if _, err := fmt.Sscanf(idStr, "%d", &id); err != nil {
			return nil, fmt.Errorf("tokenizer: added_tokens_decoder key %q is not an id", idStr)
		}
		at = append(at, added{ID: id, Content: a.Content, SingleWord: a.SingleWord, LStrip: a.LStrip, RStrip: a.RStrip, Normalized: a.Normalized, Special: a.Special})
	}
	sort.Slice(at, func(i, j int) bool { return at[i].ID < at[j].ID })
	doc := map[string]any{
		"version":      "1.0",
		"added_tokens": at,
		"normalizer":   map[string]any{"type": "NFC"},
		"pre_tokenizer": map[string]any{"type": "Sequence", "pretokenizers": []any{
			map[string]any{"type": "Split", "pattern": map[string]any{"Regex": qwen2SplitRegex}, "behavior": "Isolated", "invert": false},
			map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
		}},
		"decoder": map[string]any{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
		"model":   map[string]any{"type": "BPE", "vocab": vocab, "merges": merges, "byte_fallback": false, "ignore_merges": false},
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		return nil, err
	}
	return parseTokenizerJSON(raw, filepath.Join(dir, "vocab.json"), dir)
}

// isNotExist reports whether err means a file is simply absent.
func isNotExist(err error) bool { return errors.Is(err, fs.ErrNotExist) }

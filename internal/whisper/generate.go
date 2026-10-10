package whisper

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// GenConfig is the checkpoint's generation_config.json: the special-token ids, the language and task tables and the two suppression lists transformers' generate applies.
type GenConfig struct {
	DecoderStart  int            `json:"decoder_start_token_id"`
	EOS           int            `json:"eos_token_id"`
	NoTimestamps  int            `json:"no_timestamps_token_id"`
	MaxLength     int            `json:"max_length"`
	LangToID      map[string]int `json:"lang_to_id"`
	TaskToID      map[string]int `json:"task_to_id"`
	Suppress      []int          `json:"suppress_tokens"`
	BeginSuppress []int          `json:"begin_suppress_tokens"`
}

// LoadGenConfig reads dir/generation_config.json.
func LoadGenConfig(dir string) (GenConfig, error) {
	var g GenConfig
	raw, err := os.ReadFile(filepath.Join(dir, "generation_config.json"))
	if err != nil {
		return g, err
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		return g, fmt.Errorf("whisper: %s/generation_config.json: %w", dir, err)
	}
	return g, nil
}

// langIDs is the language tokens' ids, ascending.
func (g GenConfig) langIDs() []int {
	ids := make([]int, 0, len(g.LangToID))
	for _, id := range g.LangToID {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}

// LanguageToken is the token text of a language code: "en" is "<|en|>".
func LanguageToken(code string) string { return "<|" + strings.Trim(code, "<|>") + "|>" }

// Detect is transformers' detect_language: the decoder over the start token alone, and the language token with the highest logit. It returns the token's text ("<|en|>") and id.
func (d *Decoder) Detect(enc []float32, g GenConfig) (string, int, error) {
	ids := g.langIDs()
	if len(ids) == 0 {
		return "", 0, fmt.Errorf("whisper: generation_config.json has no lang_to_id: this checkpoint is not multilingual")
	}
	st, err := d.NewState(enc)
	if err != nil {
		return "", 0, err
	}
	l, err := st.Forward([]int{g.DecoderStart}, false)
	if err != nil {
		return "", 0, err
	}
	best := ids[0]
	for _, id := range ids {
		if l[0][id] > l[0][best] {
			best = id
		}
	}
	for tok, id := range g.LangToID {
		if id == best {
			return tok, id, nil
		}
	}
	return "", best, nil
}

// Result is one window's transcription: the prompt it was generated from, the generated ids (the stop token included when the model ended the text itself) and the language.
type Result struct {
	Prompt   []int
	IDs      []int
	Language string // the language token's text, "<|en|>"
	Stopped  bool   // ended on the stop token, not on the length cap
}

// Generate is transformers' short-form greedy generate with return_timestamps off: the prompt <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|> (the language detected when language is
// empty), then the argmax of the logits after suppress_tokens (always) and begin_suppress_tokens (at the first generated position), until the stop token or max_length tokens in all.
func (d *Decoder) Generate(enc []float32, g GenConfig, language, task string) (*Result, error) {
	var langTok string
	var langID int
	var err error
	if language == "" {
		if langTok, langID, err = d.Detect(enc, g); err != nil {
			return nil, err
		}
	} else {
		langTok = LanguageToken(language)
		id, ok := g.LangToID[langTok]
		if !ok {
			return nil, fmt.Errorf("whisper: language %q is not one of this checkpoint's %d", language, len(g.LangToID))
		}
		langID = id
	}
	if task == "" {
		task = "transcribe"
	}
	taskID, ok := g.TaskToID[task]
	if !ok {
		return nil, fmt.Errorf("whisper: task %q is not in this checkpoint's task_to_id", task)
	}
	prompt := []int{g.DecoderStart, langID, taskID, g.NoTimestamps}
	maxLen := g.MaxLength
	if maxLen <= 0 || maxLen > d.Cfg.MaxTarget {
		maxLen = d.Cfg.MaxTarget
	}
	st, err := d.NewState(enc)
	if err != nil {
		return nil, err
	}
	logits, err := st.Forward(prompt, false)
	if err != nil {
		return nil, err
	}
	res := &Result{Prompt: prompt, Language: langTok}
	for len(prompt)+len(res.IDs) < maxLen {
		l := logits[0]
		for _, id := range g.Suppress {
			if id >= 0 && id < len(l) {
				l[id] = float32(math.Inf(-1))
			}
		}
		if len(res.IDs) == 0 {
			for _, id := range g.BeginSuppress {
				if id >= 0 && id < len(l) {
					l[id] = float32(math.Inf(-1))
				}
			}
		}
		best := 0
		for i, v := range l {
			if v > l[best] {
				best = i
			}
		}
		res.IDs = append(res.IDs, best)
		if best == g.EOS {
			res.Stopped = true
			break
		}
		if logits, err = st.Forward([]int{best}, false); err != nil {
			return nil, err
		}
	}
	return res, nil
}

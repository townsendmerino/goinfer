package whisper

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
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
	MaxInitialTS  *int           `json:"max_initial_timestamp_index"`
	PrevSOT       *int           `json:"prev_sot_token_id"`
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

// planted defects of timestamps_test.go: the zero value is the correct rules.
const (
	tsNone          = iota
	tsNoPairRule    // the pair rule ignores the penultimate token
	tsNoMonotonic   // timestamps may decrease
	tsNoInitialRule // no restriction at the first generated position
	tsNoLogprobRule // the "timestamps outweigh any text token" rule off
	tsSuppressAfter // suppress_tokens run after the timestamp processor
	tsAdvanceWindow // (long form) every window advances by 30 s
	tsSingleAsPair  // (long form) a single timestamp ending treated as a pair
	tsNoTimeOffset  // (long form) the window's time offset not added
)

// planted conditioning defects of policy_test.go: the zero value is the correct prompt.
const (
	condNone             = iota
	condNoCut            // the whole previous text, uncut
	condNoDoubleTrim     // a segment's double ending timestamp not trimmed
	condNoStartOfPrev    // no <|startofprev|> in front
	polNoSpeechProcessed // the no-speech probability read from the processed scores, not the raw logits
	polSkipIgnored       // a silent window is not skipped
	polAvgNoStop         // the stop token's log-probability left out of the sum
)

// maxLength is generate's cap on the whole sequence (prompt included): max_length raised by the prompt's length (at most max_target_positions/2 - 1 of it), and never past max_target_positions
// (generation_whisper.py, _set_max_new_tokens_and_length).
func (d *Decoder) maxLength(g GenConfig, promptLen int) int {
	if g.MaxLength <= 0 {
		return d.Cfg.MaxTarget
	}
	return min(g.MaxLength+min(d.Cfg.MaxTarget/2-1, promptLen), d.Cfg.MaxTarget)
}

// promptFor is the decoder's prompt: <|startoftranscript|> <|lang|> <|task|>, then <|notimestamps|> unless timestamps are wanted. With no language it is detected from enc.
func (d *Decoder) promptFor(enc []float32, g GenConfig, language, task string, timestamps bool) (prompt []int, langTok string, err error) {
	var langID int
	if language == "" {
		if langTok, langID, err = d.Detect(enc, g); err != nil {
			return nil, "", err
		}
	} else {
		langTok = LanguageToken(language)
		id, ok := g.LangToID[langTok]
		if !ok {
			return nil, "", fmt.Errorf("whisper: language %q is not one of this checkpoint's %d", language, len(g.LangToID))
		}
		langID = id
	}
	if task == "" {
		task = "transcribe"
	}
	taskID, ok := g.TaskToID[task]
	if !ok {
		return nil, "", fmt.Errorf("whisper: task %q is not in this checkpoint's task_to_id", task)
	}
	prompt = []int{g.DecoderStart, langID, taskID}
	if !timestamps {
		prompt = append(prompt, g.NoTimestamps)
	}
	return prompt, langTok, nil
}

// Generate is transformers' short-form greedy generate with return_timestamps off: the prompt <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|> (the language detected when language is
// empty), then the argmax of the logits after begin_suppress_tokens (at the first generated position) and suppress_tokens, until the stop token or max_length tokens in all.
func (d *Decoder) Generate(enc []float32, g GenConfig, language, task string) (*Result, error) {
	prompt, langTok, err := d.promptFor(enc, g, language, task, false)
	if err != nil {
		return nil, err
	}
	a, err := d.generate(enc, g, prompt, false, 0, nil, -1)
	if err != nil {
		return nil, err
	}
	return &Result{Prompt: prompt, IDs: a.ids, Language: langTok, Stopped: a.stopped}, nil
}

// attempt is one decoding of one window at one temperature.
type attempt struct {
	ids          []int     // generated ids, the stop token included when the model ended the text itself
	stopped      bool      // ended on the stop token
	sumLogprob   float64   // sum over ids of the log-probability of the chosen token under the scores the generation produced
	lps          []float64 // each generated token's log-probability (the stop token's included)
	noSpeechProb float64   // the no-speech probability at the <|startoftranscript|> position (raw logits), when asked for
}

// avgLogprob is _retrieve_avg_logprobs: the sum over the generated tokens (the stop token counted) over their number.
func (a attempt) avgLogprob() float64 { return a.sumLogprob / float64(len(a.ids)) }

// topK is the generation config's default top-k, which a sampling generate applies after the temperature.
const topK = 50

// generate runs the loop from prompt, greedy at temperature 0 and sampled above it. The processors run in transformers' order, each on the previous one's output: begin_suppress_tokens (the first
// generated position only), suppress_tokens, and with timestamps the WhisperTimeStampLogitsProcessor; a sampled token then comes from those scores divided by the temperature, cut to the top-k, softmaxed
// and drawn with rng. sotAt is the index of <|startoftranscript|> in prompt, where the no-speech probability is read (-1: not asked for).
func (d *Decoder) generate(enc []float32, g GenConfig, prompt []int, timestamps bool, temperature float64, rng *rand.Rand, sotAt int) (attempt, error) {
	var out attempt
	st, err := d.NewState(enc)
	if err != nil {
		return out, err
	}
	var logits [][]float32
	if sotAt >= 0 {
		// the raw logits at the start token give the no-speech probability, before any processor
		first, err := st.Forward(prompt[:sotAt+1], false)
		if err != nil {
			return out, err
		}
		out.noSpeechProb = softmaxAt(first[0], g.NoTimestamps-1)
		if sotAt+1 < len(prompt) {
			if logits, err = st.Forward(prompt[sotAt+1:], false); err != nil {
				return out, err
			}
		} else {
			logits = first
		}
	} else if logits, err = st.Forward(prompt, false); err != nil {
		return out, err
	}
	maxLen := d.maxLength(g, len(prompt))
	for len(prompt)+len(out.ids) < maxLen {
		l := logits[0]
		d.processLogits(l, out.ids, g, timestamps)
		if d.condDefect == polNoSpeechProcessed && sotAt >= 0 && len(out.ids) == 0 {
			out.noSpeechProb = softmaxAt(l, g.NoTimestamps-1)
		}
		if temperature > 0 {
			sampleRescale(l, temperature)
		}
		var best int
		if temperature > 0 {
			best = sampleFrom(l, temperature, rng)
		} else {
			for i, v := range l {
				if v > l[best] {
					best = i
				}
			}
		}
		lp := logSoftmaxAt(l, best)
		out.lps = append(out.lps, lp)
		if !(best == g.EOS && d.condDefect == polAvgNoStop) {
			out.sumLogprob += lp
		}
		out.ids = append(out.ids, best)
		if best == g.EOS {
			out.stopped = true
			return out, nil
		}
		if logits, err = st.Forward([]int{best}, false); err != nil {
			return out, err
		}
	}
	return out, nil
}

// sampleRescale is transformers' TopKLogitsWarper on the processed scores l (the TemperatureLogitsWarper before it only divides every score, so the ranking is the same): the entries outside the top-k
// become -inf, the rest keep their processed value. _retrieve_avg_logprobs multiplies the generation's scores back by the temperature before its log-softmax, so the log-probabilities it reads are those of
// the processed scores restricted to the top-k set, which is what l holds afterwards; the draw is from softmax(l / temperature).
func sampleRescale(l []float32, temperature float64) {
	if len(l) <= topK {
		return
	}
	idx := make([]int, len(l))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return l[idx[a]] > l[idx[b]] })
	kept := make(map[int]bool, topK)
	for _, i := range idx[:topK] {
		kept[i] = true
	}
	neg := float32(math.Inf(-1))
	for i := range l {
		if !kept[i] {
			l[i] = neg
		}
	}
}

// sampleFrom draws an index from softmax(l / temperature), where l carries -inf outside the kept set (sampleRescale).
func sampleFrom(l []float32, temperature float64, rng *rand.Rand) int {
	mx := float32(math.Inf(-1))
	for _, v := range l {
		if v > mx {
			mx = v
		}
	}
	var z float64
	p := make([]float64, len(l))
	for i, v := range l {
		if !math.IsInf(float64(v), -1) {
			p[i] = math.Exp(float64(v-mx) / temperature)
			z += p[i]
		}
	}
	u := rng.Float64() * z
	var c float64
	for i, pi := range p {
		c += pi
		if u < c {
			return i
		}
	}
	return len(l) - 1
}

func softmaxAt(l []float32, i int) float64 {
	mx := float32(math.Inf(-1))
	for _, v := range l {
		if v > mx {
			mx = v
		}
	}
	var z float64
	for _, v := range l {
		z += math.Exp(float64(v - mx))
	}
	return math.Exp(float64(l[i]-mx)) / z
}

// logSoftmaxAt is the log-probability of index i under softmax(l), l carrying -inf for masked entries.
func logSoftmaxAt(l []float32, i int) float64 {
	mx := float32(math.Inf(-1))
	for _, v := range l {
		if v > mx {
			mx = v
		}
	}
	var z float64
	for _, v := range l {
		if !math.IsInf(float64(v), -1) {
			z += math.Exp(float64(v - mx))
		}
	}
	return float64(l[i]-mx) - math.Log(z)
}

// processLogits applies transformers' logits processors to l, the scores for the token after ids (the tokens generated so far), in its order: begin_suppress_tokens (the first generated position
// only), suppress_tokens, and with timestamps WhisperTimeStampLogitsProcessor. The order matters: the timestamp rule compares the timestamps' summed probability with the best text token's, and a
// suppressed token must not count as a text token.
func (d *Decoder) processLogits(l []float32, ids []int, g GenConfig, timestamps bool) {
	neg := float32(math.Inf(-1))
	suppress := func() {
		for _, id := range g.Suppress {
			if id >= 0 && id < len(l) {
				l[id] = neg
			}
		}
	}
	if len(ids) == 0 {
		for _, id := range g.BeginSuppress {
			if id >= 0 && id < len(l) {
				l[id] = neg
			}
		}
	}
	if d.tsDefect != tsSuppressAfter {
		suppress()
	}
	if timestamps {
		d.timestampRules(l, ids, g)
	}
	if d.tsDefect == tsSuppressAfter {
		suppress()
	}
}

// timestampRules is WhisperTimeStampLogitsProcessor over l (the scores for the next token) given the tokens generated so far (generation_whisper's begin_index is the prompt's length, so seq is what the
// model wrote). timestamp_begin is <|notimestamps|> + 1.
func (d *Decoder) timestampRules(l []float32, seq []int, g GenConfig) {
	neg := float32(math.Inf(-1))
	tsBegin := g.NoTimestamps + 1
	l[g.NoTimestamps] = neg
	lastWasTS := len(seq) >= 1 && seq[len(seq)-1] >= tsBegin
	penultimateWasTS := len(seq) < 2 || seq[len(seq)-2] >= tsBegin
	if d.tsDefect == tsNoPairRule {
		penultimateWasTS = false
	}
	if lastWasTS {
		if penultimateWasTS {
			for i := tsBegin; i < len(l); i++ {
				l[i] = neg
			}
		} else {
			for i := 0; i < g.EOS && i < len(l); i++ {
				l[i] = neg
			}
		}
	}
	lastTS, any := 0, false
	for _, t := range seq {
		if t >= tsBegin {
			lastTS, any = t, true
		}
	}
	if any && d.tsDefect != tsNoMonotonic {
		bound := lastTS + 1 // avoid <|0.00|> again
		if lastWasTS && !penultimateWasTS {
			bound = lastTS
		}
		for i := tsBegin; i < bound && i < len(l); i++ {
			l[i] = neg
		}
	}
	if len(seq) == 0 && d.tsDefect != tsNoInitialRule {
		for i := 0; i < tsBegin; i++ {
			l[i] = neg
		}
		if g.MaxInitialTS != nil {
			for i := tsBegin + *g.MaxInitialTS + 1; i < len(l); i++ {
				l[i] = neg
			}
		}
	}
	if d.tsDefect == tsNoLogprobRule {
		return
	}
	// if the summed probability of the timestamps outweighs any single text token, only a timestamp may be written
	mx := float64(math.Inf(-1))
	for _, v := range l {
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	var z float64
	for _, v := range l {
		z += math.Exp(float64(v) - mx)
	}
	lse := mx + math.Log(z)
	var tsSum float64
	for _, v := range l[tsBegin:] {
		tsSum += math.Exp(float64(v) - lse)
	}
	maxText := math.Inf(-1)
	for _, v := range l[:tsBegin] {
		if lp := float64(v) - lse; lp > maxText {
			maxText = lp
		}
	}
	if math.Log(tsSum) > maxText {
		for i := 0; i < tsBegin; i++ {
			l[i] = neg
		}
	}
}

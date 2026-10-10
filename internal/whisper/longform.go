package whisper

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/townsendmerino/aikit/audio"
)

// Segment is a stretch of text with its times in seconds from the start of the clip. Tokens are the generated ids that make it up, timestamp tokens included.
type Segment struct {
	Start, End float64
	Tokens     []int
	// The window this segment came from and the attempt that was kept for it (what verbose_json reports): the window's first feature frame, the temperature, the average log-probability,
	// the compression ratio (0 unless the threshold was set) and the no-speech probability. Not part of what transformers returns, so not compared against it.
	Seek             int
	Temperature      float64
	AvgLogprob       float64
	CompressionRatio float64
	NoSpeechProb     float64
}

const (
	timePrecision         = 0.02 // seconds per timestamp step
	timePrecisionFeatures = 0.01 // seconds per feature frame
	inputStride           = 2    // feature frames per encoder frame
)

// retrieveSegments is generation_whisper's _retrieve_segment: the generated tokens of one window (the stop token already removed) cut into segments at every pair of adjacent timestamps, and how far
// the window advances. timeOffset is the window's start in seconds, numFrames the feature frames the window really holds.
func retrieveSegments(seq []int, tsBegin int, timeOffset float64, numFrames int, defect int) (segs []Segment, advance int) {
	isTS := make([]bool, len(seq))
	for i, t := range seq {
		isTS[i] = t >= tsBegin
	}
	single := len(seq) >= 2 && !isTS[len(seq)-2] && isTS[len(seq)-1]
	if defect == tsSingleAsPair {
		single = false
	}
	var slices []int
	for i := 0; i+1 < len(seq); i++ {
		if isTS[i] && isTS[i+1] {
			slices = append(slices, i+1)
		}
	}
	if len(slices) > 0 {
		if single {
			slices = append(slices, len(seq))
		} else {
			slices[len(slices)-1]++
		}
		last := 0
		for i, cur := range slices {
			isLast := i == len(slices)-1
			sl := seq[last:cur]
			start := float64(sl[0] - tsBegin)
			endTok := sl[len(sl)-1]
			if isLast && !single {
				endTok = sl[len(sl)-2]
			}
			end := float64(endTok - tsBegin)
			segs = append(segs, Segment{Start: timeOffset + start*timePrecision, End: timeOffset + end*timePrecision, Tokens: sl})
			last = cur
		}
		if single {
			return segs, numFrames
		}
		return segs, (seq[last-2] - tsBegin) * inputStride
	}
	// no pair of timestamps: the whole generation is one segment. torch computes this in float32 (a tensor times a Python float).
	lastPos := float64(int(float32(float64(float32(float64(numFrames)*timePrecisionFeatures)) / timePrecision)))
	var ts []int
	for _, t := range seq {
		if t >= tsBegin {
			ts = append(ts, t)
		}
	}
	if len(ts) > 0 && ts[len(ts)-1] != tsBegin {
		lastPos = float64(ts[len(ts)-1] - tsBegin)
	}
	return []Segment{{Start: timeOffset, End: timeOffset + lastPos*timePrecision, Tokens: seq}}, numFrames
}

// LongResult is a timestamped transcription of a clip of any length.
type LongResult struct {
	Language string
	Segments []Segment
	Tokens   []int // every segment's tokens, in order: transformers' returned sequence without its prompt
	Windows  []WindowInfo
}

// WindowInfo records what happened to one window: which attempt's result was kept, the two decision statistics of that attempt, whether it was skipped as silence, and the prompt's length.
type WindowInfo struct {
	Seek             int
	Attempts         []float64 // the temperatures tried, in order
	CompressionRatio float64   // of the kept attempt (0 when the threshold is not set)
	AvgLogprob       float64
	NoSpeechProb     float64
	Skipped          bool
	PromptLen        int
}

// TranscribeOptions are generate's decode-policy arguments (all off in the zero value: one greedy attempt per window, nothing skipped, no conditioning).
type TranscribeOptions struct {
	Language, Task            string
	Temperatures              []float64 // tried in order for a window that needs a fallback; empty means [0]
	CompressionRatioThreshold *float64
	LogprobThreshold          *float64
	NoSpeechThreshold         *float64 // needs LogprobThreshold: a window is skipped when both its logprob is under that and its no-speech probability above this
	ConditionOnPrev           bool
	Seed                      uint64 // the sampling stream of an attempt above temperature 0
}

// compressionRatio is _retrieve_compression_ratio: each token id as int(log2(vocab)/8)+1 little-endian bytes, the length over its zlib-compressed length.
func compressionRatio(ids []int, vocab int) float64 {
	n := int(math.Log2(float64(vocab))/8) + 1
	raw := make([]byte, 0, len(ids)*n)
	for _, id := range ids {
		for k := range n {
			raw = append(raw, byte(id>>(8*k)))
		}
	}
	var buf bytes.Buffer
	zw := zlib.NewWriter(&buf)
	zw.Write(raw)
	zw.Close()
	return float64(len(raw)) / float64(buf.Len())
}

// needFallback is _need_fallback for one attempt (ids include the stop token).
func needFallback(a attempt, ratio float64, o TranscribeOptions) (needs, skip bool) {
	if o.CompressionRatioThreshold != nil && ratio > *o.CompressionRatioThreshold {
		needs = true
	}
	if o.LogprobThreshold != nil && a.avgLogprob() < *o.LogprobThreshold {
		needs = true
	}
	if o.NoSpeechThreshold != nil && o.LogprobThreshold != nil && a.avgLogprob() < *o.LogprobThreshold && a.noSpeechProb > *o.NoSpeechThreshold {
		return false, true
	}
	return needs, false
}

// TranscribeTimestamps is TranscribeWith with the zero options: transformers' generate(return_timestamps=True, return_segments=True) with nothing else set.
func (t *Transcriber) TranscribeTimestamps(samples []float32, language, task string) (*LongResult, error) {
	return t.TranscribeWith(samples, TranscribeOptions{Language: language, Task: task})
}

// TranscribeWith is transformers' generate(return_timestamps=True, return_segments=True, ...) for one clip: the sequential loop over a clip's features, each window from the previous segment's end. A clip up
// to 30 s runs it over its 30 s-padded features (3000 frames, the short form's: after a window that did not end on a single timestamp the next window starts at the last timestamp, as transformers' does); a
// longer one over its unpadded features. The language is detected once, on the first window, when none is given. A window that needs a fallback is decoded again at the next temperature; the last
// temperature's result is kept; a window that is silent by both thresholds is skipped; with ConditionOnPrev each window after the first is preceded by the earlier text.
func (t *Transcriber) TranscribeWith(samples []float32, o TranscribeOptions) (*LongResult, error) {
	g, d := t.Gen, t.Dec
	if o.NoSpeechThreshold != nil && o.LogprobThreshold == nil {
		return nil, fmt.Errorf("whisper: no_speech_threshold needs logprob_threshold (a window is skipped only when its logprob is under it too)")
	}
	temps := o.Temperatures
	if len(temps) == 0 {
		temps = []float64{0}
	}
	rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
	var feats []float32
	var T int
	if len(samples) <= MaxSamples {
		f, _, err := audio.WhisperFeatures(samples, t.Mels)
		if err != nil {
			return nil, err
		}
		feats, T = f, WindowFrames
	} else {
		f, n, err := longestFeatures(samples, t.Mels, featNone)
		if err != nil {
			return nil, err
		}
		feats, T = f, n
	}
	tsBegin := g.NoTimestamps + 1
	prevSOT := -1
	if g.PrevSOT != nil {
		prevSOT = *g.PrevSOT
	} else if len(g.Suppress) >= 2 {
		prevSOT = g.Suppress[len(g.Suppress)-2]
	}
	res := &LongResult{}
	var init []int
	doCond := o.ConditionOnPrev
	for seek := 0; seek < T; {
		num := min(T-seek, WindowFrames)
		enc, err := t.Enc.Forward(Window(feats, t.Mels, T, seek, num)) // a short clip's padded 3000 frames are stepped through by seek like any other: the loop is the same
		if err != nil {
			return nil, err
		}
		if init == nil {
			if init, res.Language, err = d.promptFor(enc, g, o.Language, o.Task, true); err != nil {
				return nil, err
			}
		}
		prompt, sotAt := init, 0
		if doCond && len(res.Segments) > 0 && prevSOT >= 0 {
			prompt, sotAt = d.conditionedPrompt(res.Segments, init, prevSOT, tsBegin)
		}
		info := WindowInfo{Seek: seek, PromptLen: len(prompt)}
		var kept attempt
		var needs, skip bool
		var lastTemp float64
		for i, temp := range temps {
			a, err := d.generate(enc, g, prompt, true, temp, rng, sotAt)
			if err != nil {
				return nil, err
			}
			ratio := 0.0
			if o.CompressionRatioThreshold != nil {
				ratio = compressionRatio(a.ids, d.Cfg.Vocab)
			}
			needs, skip = needFallback(a, ratio, o)
			if d.condDefect == polSkipIgnored {
				skip = false
			}
			kept, lastTemp = a, temp
			info.Attempts = append(info.Attempts, temp)
			info.CompressionRatio, info.AvgLogprob, info.NoSpeechProb = ratio, a.avgLogprob(), a.noSpeechProb
			if !needs || i == len(temps)-1 || skip {
				break
			}
		}
		doCond = o.ConditionOnPrev && lastTemp < 0.5
		if skip {
			info.Skipped = true
			res.Windows = append(res.Windows, info)
			seek += num
			continue
		}
		ids := kept.ids
		if len(ids) > 0 && ids[len(ids)-1] == g.EOS {
			ids = ids[:len(ids)-1]
		}
		if len(ids) == 0 {
			return nil, fmt.Errorf("whisper: window at frame %d generated no tokens", seek)
		}
		offset := float64(seek) * timePrecision / inputStride
		if d.tsDefect == tsNoTimeOffset {
			offset = 0
		}
		segs, adv := retrieveSegments(ids, tsBegin, offset, num, d.tsDefect)
		if d.tsDefect == tsAdvanceWindow {
			adv = num
		}
		if adv <= 0 {
			return nil, fmt.Errorf("whisper: window at frame %d did not advance", seek)
		}
		res.Windows = append(res.Windows, info)
		for i := range segs {
			segs[i].Seek, segs[i].Temperature, segs[i].AvgLogprob, segs[i].CompressionRatio, segs[i].NoSpeechProb = seek, lastTemp, info.AvgLogprob, info.CompressionRatio, info.NoSpeechProb
		}
		seek += adv
		res.Segments = append(res.Segments, segs...)
		for _, s := range segs {
			res.Tokens = append(res.Tokens, s.Tokens...)
		}
	}
	return res, nil
}

// conditionedPrompt is _prepare_decoder_input_ids' conditioned form: <|startofprev|>, the earlier segments' tokens (a segment that ends on two timestamps drops its last token) cut to the last
// max_target_positions/2 - 1, then the window's own prompt. sotAt is where <|startoftranscript|> now sits.
func (d *Decoder) conditionedPrompt(segs []Segment, init []int, prevSOT, tsBegin int) (prompt []int, sotAt int) {
	var prev []int
	for _, s := range segs {
		tok := s.Tokens
		if d.condDefect != condNoDoubleTrim && len(tok) > 2 && tok[len(tok)-2] >= tsBegin {
			tok = tok[:len(tok)-1]
		}
		prev = append(prev, tok...)
	}
	if cut := d.Cfg.MaxTarget/2 - 1; d.condDefect != condNoCut && len(prev) > cut {
		prev = prev[len(prev)-cut:]
	}
	prompt = append(prompt, prevSOT)
	if d.condDefect == condNoStartOfPrev {
		prompt = prompt[:0]
	}
	prompt = append(prompt, prev...)
	sotAt = len(prompt)
	return append(prompt, init...), sotAt
}

// TimestampText decodes a LongResult's tokens to text, skipping every special token and the timestamps.
func (t *Transcriber) TimestampText(r *LongResult) string { return t.Text(r.Tokens) }

package whisper

import (
	"fmt"

	"github.com/townsendmerino/aikit/audio"
)

// Segment is a stretch of text with its times in seconds from the start of the clip. Tokens are the generated ids that make it up, timestamp tokens included.
type Segment struct {
	Start, End float64
	Tokens     []int
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
}

// TranscribeTimestamps is transformers' generate(return_timestamps=True, return_segments=True) for one clip, greedy: the sequential loop over a clip's features, each window from the previous segment's end. A clip up
// to 30 s runs it over its 30 s-padded features (3000 frames, the short form's: after a window that did not end on a single timestamp the next window starts at the last timestamp, as transformers' does); a longer
// one over its unpadded features. The language is detected once, on the first window, when none is given.
func (t *Transcriber) TranscribeTimestamps(samples []float32, language, task string) (*LongResult, error) {
	g, d := t.Gen, t.Dec
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
	res := &LongResult{}
	var prompt []int
	for seek := 0; seek < T; {
		num := min(T-seek, WindowFrames)
		win := Window(feats, t.Mels, T, seek, num) // a short clip's padded 3000 frames are stepped through by seek like any other: the loop is the same
		enc, err := t.Enc.Forward(win)
		if err != nil {
			return nil, err
		}
		if prompt == nil {
			if prompt, res.Language, err = d.promptFor(enc, g, language, task, true); err != nil {
				return nil, err
			}
		}
		ids, _, err := d.generate(enc, g, prompt, true)
		if err != nil {
			return nil, err
		}
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
		seek += adv
		res.Segments = append(res.Segments, segs...)
		for _, s := range segs {
			res.Tokens = append(res.Tokens, s.Tokens...)
		}
	}
	return res, nil
}

// TimestampText decodes a LongResult's tokens to text, skipping every special token and the timestamps.
func (t *Transcriber) TimestampText(r *LongResult) string { return t.Text(r.Tokens) }

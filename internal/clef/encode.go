// Package clef is the Go side of Cloudflare's Clef-flash joint decision model (Route C of docs/tasks/task-constrained-confidence.md,
// D10-D14): the record encoder here (D12), the joint head beside it.
//
// The encoder is a port of encode_record in the checkpoint's own joint_schema_model.py, read in full for
// docs/measurements/decisions-d10-clef-2026-10-02.md. It turns {state, questions} into ONE token sequence
// (prefix + state + schema + suffix) plus, per question, the token spans the head pools over. Its contract is input-identity with the
// reference: the same prompt text fragments, the same token ids, the same spans, on the 150 recorded items (encode_test.go).
package clef

import (
	"errors"
	"fmt"
	"sort"
)

// DefaultMaxLength is the reference's max_length. The blog claims a 64k window; the code's default is this.
const DefaultMaxLength = 16384

const (
	systemPrompt = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options."
	prefixText   = "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n<|im_start|>user\nSTATE:\n"
	suffixText   = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"

	defaultTrue  = "The proposition is true or the answer is yes."
	defaultFalse = "The proposition is false or the answer is no."
)

// Question types, in the reference's QUESTION_TYPES numbering (the head's embedding index).
const (
	TypeNoul   = 0
	TypeChoice = 1
	TypeScore  = 2
)

// Tokenize turns one text fragment into token ids exactly as the reference's tokenizer(text, add_special_tokens=False).input_ids does:
// special-token text inside the fragment (including inside a customer's state) becomes the control token. The reference does that and
// the head was trained on it, so a faithful port must too; see the package doc of the caller for what that means for untrusted state.
type Tokenize func(text string) ([]int, error)

// Question is one encoded question: where its instruction and each option sit in the sequence, and the option ids in the order the
// head's logits come in (noul: true, false; choice: alphabetical by key, NOT the order sent; score: "0", "1", ...).
type Question struct {
	ID           string
	Type         int
	QuestionSpan [2]int
	OptionSpans  [][2]int
	OptionIDs    []string
}

// Encoded is encode_record's EncodedRecord, without the media half (images and videos are refused until multimodal.md P8a).
type Encoded struct {
	InputIDs  []int
	Questions []Question
}

// Encode builds the model input for a request body {"state": ..., "questions": {id: {type, instructions?, criteria?}, ...}}. The
// questions keep the order they were sent in. maxLength <= 0 means DefaultMaxLength. A state that does not fit is cut to its FIRST
// tokens; a schema that alone does not fit is an error.
func Encode(request []byte, tok Tokenize, maxLength int) (*Encoded, error) {
	if maxLength <= 0 {
		maxLength = DefaultMaxLength
	}
	root, err := parse(request)
	if err != nil {
		return nil, err
	}
	if root.k != kObject {
		return nil, errors.New("clef: the request must be a JSON object")
	}
	if imgs := root.get("images"); !imgs.falsy() {
		return nil, errors.New("clef: images are not supported on this route yet (multimodal.md P8a)")
	}
	if vids := root.get("videos"); !vids.falsy() {
		return nil, errors.New("clef: videos are not supported on this route yet (multimodal.md P8a)")
	}
	stateV := root.get("state")
	if stateV == nil {
		return nil, errors.New("clef: the request has no state")
	}
	qs := root.get("questions")
	if qs == nil || qs.k != kObject || len(qs.keys) == 0 {
		return nil, errors.New("clef: questions must be a non-empty object keyed by question id")
	}

	// Same call order as the reference, so a recording tokenizer sees the same sequence: the schema fragments, then the prefix, the
	// suffix, and the state last.
	schema, err := tok("\n\nSCHEMA FIELDS:\n")
	if err != nil {
		return nil, err
	}
	var questions []Question
	for qi, id := range qs.keys {
		q := qs.vals[qi]
		if q.k != kObject {
			return nil, fmt.Errorf("clef: question %q is not an object", id)
		}
		typeV := q.get("type")
		if typeV == nil || typeV.k != kString {
			return nil, fmt.Errorf("clef: question %q has no string type", id)
		}
		var qtype int
		switch typeV.s {
		case "noul":
			qtype = TypeNoul
		case "choice":
			qtype = TypeChoice
		case "score":
			qtype = TypeScore
		default:
			return nil, fmt.Errorf("clef: question %q has type %q, want noul, choice or score", id, typeV.s)
		}
		opts, err := questionOptions(id, typeV.s, q.get("criteria"))
		if err != nil {
			return nil, err
		}
		if schema, err = appendTokens(schema, tok, fmt.Sprintf("\nFIELD %d\nID: %s\nTYPE: %s\nINSTRUCTION: ", qi+1, id, typeV.s)); err != nil {
			return nil, err
		}
		qStart := len(schema)
		ins := q.get("instructions")
		var insText string
		if ins.isNull() || (ins.k == kString && ins.s == "") {
			insText = id
		} else if insText, err = render(ins); err != nil {
			return nil, err
		}
		if schema, err = appendTokens(schema, tok, insText); err != nil {
			return nil, err
		}
		eq := Question{ID: id, Type: qtype, QuestionSpan: [2]int{qStart, len(schema)}}
		if schema, err = appendTokens(schema, tok, "\nALLOWED OPTIONS:\n"); err != nil {
			return nil, err
		}
		for oi, o := range opts {
			if schema, err = appendTokens(schema, tok, fmt.Sprintf("OPTION %d: ", oi+1)); err != nil {
				return nil, err
			}
			start := len(schema)
			// {"option_id": id, "description": d} with sorted keys: description first, and only when d is not None.
			sem := &value{k: kObject, keys: []string{"option_id"}, vals: []*value{{k: kString, s: o.id}}}
			if !o.desc.isNull() {
				sem.keys, sem.vals = append(sem.keys, "description"), append(sem.vals, o.desc)
			}
			text, err := render(sem)
			if err != nil {
				return nil, err
			}
			if schema, err = appendTokens(schema, tok, text); err != nil {
				return nil, err
			}
			eq.OptionSpans = append(eq.OptionSpans, [2]int{start, len(schema)})
			eq.OptionIDs = append(eq.OptionIDs, o.id)
			if schema, err = appendTokens(schema, tok, "\n"); err != nil {
				return nil, err
			}
		}
		if schema, err = appendTokens(schema, tok, "END FIELD\n"); err != nil {
			return nil, err
		}
		questions = append(questions, eq)
	}

	prefix, err := tok(prefixText)
	if err != nil {
		return nil, err
	}
	suffix, err := tok(suffixText)
	if err != nil {
		return nil, err
	}
	stateText, err := render(stateV)
	if err != nil {
		return nil, err
	}
	state, err := tok(stateText)
	if err != nil {
		return nil, err
	}
	fixed := len(prefix) + len(schema) + len(suffix)
	if fixed > maxLength {
		return nil, fmt.Errorf("clef: the schema requires %d tokens before the state; the maximum is %d", fixed, maxLength)
	}
	if room := maxLength - fixed; len(state) > room {
		state = state[:room]
	}
	shift := len(prefix) + len(state)
	for i := range questions {
		questions[i].QuestionSpan[0] += shift
		questions[i].QuestionSpan[1] += shift
		for j := range questions[i].OptionSpans {
			questions[i].OptionSpans[j][0] += shift
			questions[i].OptionSpans[j][1] += shift
		}
	}
	ids := make([]int, 0, fixed+len(state))
	ids = append(append(append(append(ids, prefix...), state...), schema...), suffix...)
	return &Encoded{InputIDs: ids, Questions: questions}, nil
}

func appendTokens(dst []int, tok Tokenize, text string) ([]int, error) {
	ids, err := tok(text)
	if err != nil {
		return nil, err
	}
	return append(dst, ids...), nil
}

type option struct {
	id   string
	desc *value // nil or null: no description key
}

// questionOptions is the reference's question_options. A noul question's criteria can override the two descriptions and nothing else;
// a choice's options are sorted by key, not kept in the order sent; a score's are its list indices.
func questionOptions(id, qtype string, criteria *value) ([]option, error) {
	switch qtype {
	case "noul":
		t, f := &value{k: kString, s: defaultTrue}, &value{k: kString, s: defaultFalse}
		if !criteria.falsy() {
			if criteria.k != kObject {
				return nil, fmt.Errorf("clef: question %q: a noul question's criteria must be an object", id)
			}
			if v := criteria.get("true"); v != nil {
				t = v
			}
			if v := criteria.get("false"); v != nil {
				f = v
			}
		}
		return []option{{"true", t}, {"false", f}}, nil
	case "choice":
		if criteria == nil || criteria.k != kObject {
			return nil, fmt.Errorf("clef: question %q: a choice question needs criteria as an object of option -> description", id)
		}
		out := make([]option, len(criteria.keys))
		for i, k := range criteria.keys {
			out[i] = option{k, criteria.vals[i]}
		}
		sort.Slice(out, func(a, b int) bool { return out[a].id < out[b].id })
		return out, nil
	default:
		if criteria == nil || criteria.k != kArray {
			return nil, fmt.Errorf("clef: question %q: a score question needs criteria as a list of level descriptions", id)
		}
		out := make([]option, len(criteria.arr))
		for i, d := range criteria.arr {
			out[i] = option{fmt.Sprint(i), d}
		}
		return out, nil
	}
}

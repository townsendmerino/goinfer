package inkling2

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

// Inkling 2 message and content control markers.
const (
	User     = "<|message_user|>"
	Model    = "<|message_model|>"
	System   = "<|message_system|>"
	Tool     = "<|message_tool|>"
	Text     = "<|content_text|>"
	Thinking = "<|content_thinking|>"
	XML      = "<|content_xml|>"
	CallJSON = "<|content_invoke_tool_json|>"
	CallText = "<|content_invoke_tool_text|>"
	End      = "<|end_message|>"
	EOS      = "<|content_model_end_sampling|>"
)

// Efforts maps Inkling 2's named reasoning effort strings to the numeric effort level written into the prompt.
var Efforts = map[string]float64{
	"none":    0,
	"minimal": 0.1,
	"low":     0.2,
	"medium":  0.7,
	"high":    0.9,
	"xhigh":   0.99,
	"max":     0.99,
}

// Segments renders tool declarations, system prompt, thinking effort, and conversation turns in Inkling 2 format.
func Segments(effort, system string, turns []chatcore.Turn, tools []chatcore.Tool) []chatcore.Segment {
	var b chatcore.SegBuf
	if len(tools) > 0 {
		b.Sp(System)
		b.Ct("tool_declare")
		b.Sp(XML)
		b.Ct(toolDecls(tools))
		b.Sp(End)
	}
	if system != "" {
		b.Sp(System + Text)
		b.Ct(system)
		b.Sp(End)
	}
	b.Sp(System + Text)
	b.Ct("Thinking effort level: " + effort)
	b.Sp(End)
	for _, tr := range turns {
		switch tr.Role {
		case "assistant":
			if tr.Reasoning != "" {
				b.Sp(Model + Thinking)
				b.Ct(tr.Reasoning)
				b.Sp(End)
			}
			if tr.Content != "" || (len(tr.ToolCalls) == 0 && tr.Reasoning == "") {
				b.Sp(Model + Text)
				b.Ct(tr.Content)
				b.Sp(End)
			}
			for _, c := range tr.ToolCalls {
				b.Sp(Model)
				b.Ct(c.Name)
				b.Sp(CallJSON)
				b.Ct(callJSON(c))
				b.Sp(End)
			}
			b.Sp(EOS)
		case "tool":
			b.Sp(Tool)
			b.Ct(tr.ToolName)
			b.Sp(Text)
			b.Ct(tr.Content)
			b.Sp(End)
		default:
			b.Sp(User + Text)
			b.Ct(tr.Content)
			b.Sp(End)
		}
	}
	return b.Segs
}

func pyJSON(raw json.RawMessage, sortKeys bool) string {
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if len(bytes.TrimSpace(raw)) == 0 || dec.Decode(&v) != nil {
		return string(raw)
	}
	if !sortKeys {
		var c bytes.Buffer
		if json.Compact(&c, raw) == nil {
			return unescapeHTML(c.String())
		}
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if enc.Encode(v) != nil {
		return string(raw)
	}
	return strings.TrimSuffix(buf.String(), "\n")
}

func unescapeHTML(s string) string {
	return strings.NewReplacer(`<`, "<", `>`, ">", `&`, "&").Replace(s)
}

func toolDecls(tools []chatcore.Tool) string {
	specs := make([]map[string]json.RawMessage, len(tools))
	for i, tl := range tools {
		params := tl.Parameters
		if len(bytes.TrimSpace(params)) == 0 || string(bytes.TrimSpace(params)) == "null" {
			params = json.RawMessage("{}")
		}
		d, _ := json.Marshal(tl.Description)
		n, _ := json.Marshal(tl.Name)
		specs[i] = map[string]json.RawMessage{"description": d, "name": n, "parameters": params, "type": json.RawMessage(`"function"`)}
	}
	raw, _ := json.Marshal(specs)
	return pyJSON(raw, true)
}

func callJSON(c chatcore.ToolCall) string {
	var args map[string]json.RawMessage
	dec := json.NewDecoder(bytes.NewReader(c.Arguments))
	dec.UseNumber()
	_ = dec.Decode(&args)
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	name, _ := json.Marshal(c.Name)
	var b strings.Builder
	b.WriteString(`{"name":` + pyJSON(name, true) + `,"args":{`)
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		kj, _ := json.Marshal(k)
		b.WriteString(pyJSON(kj, false) + ":" + pyJSON(args[k], false))
	}
	b.WriteString("}}")
	return b.String()
}

type route uint8

const (
	toContent route = iota
	toReasoning
	toCall
)

// Call holds a raw tool invocation emitted by an Inkling 2 reply.
type Call struct {
	Name, Body string
}

// Splitter incrementally separates a streamed Inkling 2 reply into reasoning, content, and tool calls.
type Splitter struct {
	buf      string
	state    uint8
	header   string
	route    route
	emitted  [3]bool
	started  bool
	reasoned bool
	answered bool
	done     bool
	calls    []Call
	callBody strings.Builder
}

// NewSplitter returns a fresh Inkling 2 reply stream splitter.
func NewSplitter() *Splitter { return &Splitter{} }

// Calls returns the raw tool invocation messages collected by the splitter.
func (s *Splitter) Calls() []Call { return s.calls }

// ToolCalls converts the splitter's collected tool invocation messages into structured ToolCalls.
func (s *Splitter) ToolCalls() []chatcore.ToolCall { return ToolCalls(s.calls) }

var markers = []string{Model, Text, Thinking, CallJSON, CallText, End, EOS, User, System, Tool}

func holdback(s string) int {
	i := strings.LastIndexByte(s, '<')
	if i < 0 {
		return 0
	}
	tail := s[i:]
	for _, m := range markers {
		if len(tail) < len(m) && strings.HasPrefix(m, tail) {
			return len(tail)
		}
	}
	return 0
}

// InReasoning reports whether the reply has entered a thinking message without yet emitting answer content.
func (s *Splitter) InReasoning() bool {
	if s.state == 2 && s.route == toContent {
		return false
	}
	return s.reasoned && !s.answered
}

// Push feeds the next decoded chunk and returns reasoning and content safe to emit immediately.
func (s *Splitter) Push(chunk string) (reasoning, content string) {
	if s.done {
		return "", ""
	}
	s.buf += chunk
	var out [3]strings.Builder
	emit := func(r route, text string) {
		if text == "" {
			return
		}
		if r == toCall {
			s.callBody.WriteString(text)
			return
		}
		if !s.started {
			s.started = true
			if s.emitted[r] {
				text = "\n\n" + text
			}
			s.emitted[r] = true
		}
		if r == toContent {
			s.answered = true
		}
		out[r].WriteString(text)
	}
	for !s.done {
		i, m := firstMarker(s.buf)
		if i < 0 {
			keep := holdback(s.buf)
			text := s.buf[:len(s.buf)-keep]
			s.buf = s.buf[len(s.buf)-keep:]
			switch s.state {
			case 0:
				if strings.TrimSpace(text) != "" {
					s.route = toContent
					emit(toContent, text)
				}
			case 1:
				s.header += text
			default:
				emit(s.route, text)
			}
			break
		}
		text := s.buf[:i]
		s.buf = s.buf[i+len(m):]
		switch s.state {
		case 0:
			if strings.TrimSpace(text) != "" {
				s.route = toContent
				emit(toContent, text)
			}
		case 1:
			s.header += text
		default:
			emit(s.route, text)
		}
		switch m {
		case Model:
			s.closeMessage()
			s.state, s.header = 1, ""
		case Thinking, Text, CallJSON, CallText:
			if s.state != 1 {
				s.closeMessage()
			}
			s.state, s.started = 2, false
			switch m {
			case Thinking:
				s.route, s.reasoned = toReasoning, true
			case Text:
				s.route = toContent
			default:
				s.route = toCall
				s.callBody.Reset()
			}
		case End:
			s.closeMessage()
		case EOS, User, System, Tool:
			s.closeMessage()
			s.done = true
		}
	}
	return out[toReasoning].String(), out[toContent].String()
}

func (s *Splitter) closeMessage() {
	if s.state == 2 && s.route == toCall {
		s.calls = append(s.calls, Call{Name: strings.TrimSpace(s.header), Body: s.callBody.String()})
		s.callBody.Reset()
	}
	s.state, s.started, s.route = 0, false, toContent
}

func firstMarker(s string) (int, string) {
	best, bm := -1, ""
	for _, m := range markers {
		if i := strings.Index(s, m); i >= 0 && (best < 0 || i < best) {
			best, bm = i, m
		}
	}
	return best, bm
}

// Flush ends the reply and returns any held-back reasoning or content tail.
func (s *Splitter) Flush() (reasoning, content string) {
	rest := s.buf
	s.buf = ""
	if s.done || rest == "" {
		if !s.done {
			s.closeMessage()
		}
		return "", ""
	}
	var r, c string
	switch {
	case s.state == 2 && s.route == toReasoning:
		r = rest
	case s.state == 2 && s.route == toCall:
		s.callBody.WriteString(rest)
	case s.state == 1:
	default:
		if strings.TrimSpace(rest) != "" || s.state == 2 {
			c = rest
			s.answered = true
		}
	}
	s.closeMessage()
	return r, c
}

// ToolCalls converts raw Inkling 2 invocation messages into structured ToolCalls.
func ToolCalls(calls []Call) []chatcore.ToolCall {
	var out []chatcore.ToolCall
	for _, c := range calls {
		var body struct {
			Name string          `json:"name"`
			Args json.RawMessage `json:"args"`
		}
		raw := bytes.TrimSpace([]byte(c.Body))
		if json.Unmarshal(raw, &body) != nil {
			continue
		}
		args := body.Args
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		var obj map[string]json.RawMessage
		var compact bytes.Buffer
		if json.Unmarshal(args, &obj) != nil || obj == nil || json.Compact(&compact, args) != nil {
			continue
		}
		name := c.Name
		if name == "" {
			name = body.Name
		}
		if name == "" {
			continue
		}
		out = append(out, chatcore.ToolCall{Name: name, Arguments: json.RawMessage(compact.Bytes())})
	}
	return out
}

// ParseTools parses a complete Inkling 2 reply into structured tool calls and answer prose.
func ParseTools(out string) ([]chatcore.ToolCall, string) {
	sp := NewSplitter()
	_, c1 := sp.Push(out)
	_, c2 := sp.Flush()
	calls := sp.ToolCalls()
	if len(calls) == 0 {
		return nil, out
	}
	return calls, strings.TrimSpace(c1 + c2)
}

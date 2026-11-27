package chat

import (
	"bytes"
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// Inkling 2's markers. Every message is <|message_ROLE|>[name]<|content_KIND|>body<|end_message|>; a model turn ends with
// <|content_model_end_sampling|>, the end-of-sequence token.
const (
	ikUser     = "<|message_user|>"
	ikModel    = "<|message_model|>"
	ikSystem   = "<|message_system|>"
	ikTool     = "<|message_tool|>"
	ikText     = "<|content_text|>"
	ikThinking = "<|content_thinking|>"
	ikXML      = "<|content_xml|>"
	ikCallJSON = "<|content_invoke_tool_json|>"
	ikCallText = "<|content_invoke_tool_text|>"
	ikEnd      = "<|end_message|>"
	ikEOS      = "<|content_model_end_sampling|>"
)

// inklingEfforts maps the template's named reasoning efforts to the level it writes (effort_map in chat_template.jinja).
var inklingEfforts = map[string]float64{"none": 0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}

// Inkling2 is thinkingmachines' Inkling 2 template (chat_template.jinja): tools declared first as
// <|message_system|>tool_declare<|content_xml|>{json}<|end_message|>, the system prompt, one
// "<|message_system|><|content_text|>Thinking effort level: E<|end_message|>" before the first non-system message, then the
// turns. There is no generation prompt: the model opens its own <|message_model|>. An assistant turn is its thinking message
// (when it carries reasoning), its text message, one <|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}
// <|end_message|> per call, and <|content_model_end_sampling|>; a tool result is <|message_tool|>NAME<|content_text|>…
// <|end_message|>. The effort defaults to 0.9; WithThinking(ThinkOff) writes 0 and WithReasoningEffort a named level.
func Inkling2() *Template { return inkling2Template(-1) }

// inkling2Template is Inkling2 with the effort level fixed; a negative effort is the template's default.
func inkling2Template(effort float64) *Template {
	t := &Template{name: "inkling2", stops: []string{ikEOS}}
	e := ""
	if effort >= 0 {
		e = strconv.FormatFloat(effort, 'g', -1, 64)
	}
	return t.inklingWith(e, ThinkAsIs)
}

// inklingWith is a copy of t writing effort level effort ("" for the default) under thinking mode m.
func (t *Template) inklingWith(effort string, m ThinkMode) *Template {
	c := *t
	c.effort, c.think = effort, m
	cp := &c
	cp.render = func(system string, turns []Turn) []Segment { return cp.inkling2Segments(system, turns, nil) }
	return cp
}

// inklingEffort is the level the effort message writes: "0" when thinking is off, else the set or default level.
func (t *Template) inklingEffort() string {
	switch {
	case t.think == ThinkOff:
		return "0"
	case t.effort != "":
		return t.effort
	}
	return "0.9"
}

func (t *Template) inkling2Segments(system string, turns []Turn, tools []Tool) []Segment {
	var b segBuf
	if len(tools) > 0 {
		b.sp(ikSystem)
		b.ct("tool_declare")
		b.sp(ikXML)
		b.ct(inklingToolDecls(tools))
		b.sp(ikEnd)
	}
	if system != "" {
		b.sp(ikSystem + ikText)
		b.ct(system)
		b.sp(ikEnd)
	}
	b.sp(ikSystem + ikText)
	b.ct("Thinking effort level: " + t.inklingEffort())
	b.sp(ikEnd)
	for _, tr := range turns {
		switch tr.Role {
		case "assistant":
			if tr.Reasoning != "" {
				b.sp(ikModel + ikThinking)
				b.ct(tr.Reasoning)
				b.sp(ikEnd)
			}
			if tr.Content != "" || (len(tr.ToolCalls) == 0 && tr.Reasoning == "") {
				b.sp(ikModel + ikText)
				b.ct(tr.Content)
				b.sp(ikEnd)
			}
			for _, c := range tr.ToolCalls {
				b.sp(ikModel)
				b.ct(c.Name)
				b.sp(ikCallJSON)
				b.ct(inklingCallJSON(c))
				b.sp(ikEnd)
			}
			b.sp(ikEOS)
		case "tool":
			b.sp(ikTool)
			b.ct(tr.ToolName)
			b.sp(ikText)
			b.ct(tr.Content)
			b.sp(ikEnd)
		default:
			b.sp(ikUser + ikText)
			b.ct(tr.Content)
			b.sp(ikEnd)
		}
	}
	return b.segs
}

// pyJSON writes v as Python's json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=sort): no HTML escaping, and
// sorted object keys at every level when sort is set (Go's encoder sorts map keys, so a sorted value decodes into maps).
func pyJSON(raw json.RawMessage, sort bool) string {
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if len(bytes.TrimSpace(raw)) == 0 || dec.Decode(&v) != nil {
		return string(raw)
	}
	if !sort {
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

// unescapeHTML undoes json.Compact's keeping of the input's own escapes: it does not add any, so this is the identity unless
// the caller's JSON spelled <, > or & as <, >, &, which Python's dumps would print raw.
func unescapeHTML(s string) string {
	return strings.NewReplacer(`<`, "<", `>`, ">", `&`, "&").Replace(s)
}

// inklingToolDecls is the tool_declare body: the tool specs as one JSON list with sorted keys, compact.
func inklingToolDecls(tools []Tool) string {
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

// inklingCallJSON is a call's body as the template writes it: {"name":NAME,"args":{…}} with the argument keys sorted (case
// sensitive) and each value compact in its own key order.
func inklingCallJSON(c ToolCall) string {
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

// inklingCall is a tool call the model wrote: the name after <|message_model|> and the JSON body.
type inklingCall struct{ name, body string }

type hmRoute uint8

const (
	hmToContent hmRoute = iota
	hmToReasoning
	hmToCall
)

// inklingSplitter reads an Inkling 2 reply, a sequence of model messages: <|content_thinking|> to the reasoning,
// <|content_text|> to the answer, and an invoke message to the calls, shown in neither. Text outside any message is answer,
// so a reply that does not speak the format degrades to plain text. The end-of-sequence marker is a stop id and never in the
// text; seen as text it ends the reply. Any partition of a reply into Pushes gives the same output as one Push.
type inklingSplitter struct {
	buf      string
	state    uint8 // 0 between messages, 1 in a header (after <|message_model|>), 2 in a body
	header   string
	route    hmRoute
	emitted  [3]bool
	started  bool
	reasoned bool
	answered bool
	done     bool
	calls    []inklingCall
	callBody strings.Builder
}

var ikMarkers = []string{ikModel, ikText, ikThinking, ikCallJSON, ikCallText, ikEnd, ikEOS, ikUser, ikSystem, ikTool}

// holdback is how many trailing bytes of s could begin a marker.
func ikHoldback(s string) int {
	i := strings.LastIndexByte(s, '<')
	if i < 0 {
		return 0
	}
	tail := s[i:]
	for _, m := range ikMarkers {
		if len(tail) < len(m) && strings.HasPrefix(m, tail) {
			return len(tail)
		}
	}
	return 0
}

func (s *inklingSplitter) InReasoning() bool {
	if s.state == 2 && s.route == hmToContent {
		return false
	}
	return s.reasoned && !s.answered
}

func (s *inklingSplitter) Push(chunk string) (reasoning, content string) {
	if s.done {
		return "", ""
	}
	s.buf += chunk
	var out [3]strings.Builder
	emit := func(r hmRoute, text string) {
		if text == "" {
			return
		}
		if r == hmToCall {
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
		if r == hmToContent {
			s.answered = true
		}
		out[r].WriteString(text)
	}
	for !s.done {
		i, m := firstMarker(s.buf)
		if i < 0 {
			keep := ikHoldback(s.buf)
			text := s.buf[:len(s.buf)-keep]
			s.buf = s.buf[len(s.buf)-keep:]
			switch s.state {
			case 0:
				if strings.TrimSpace(text) != "" {
					s.route = hmToContent
					emit(hmToContent, text)
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
				s.route = hmToContent
				emit(hmToContent, text)
			}
		case 1:
			s.header += text
		default:
			emit(s.route, text)
		}
		switch m {
		case ikModel:
			s.closeMessage()
			s.state, s.header = 1, ""
		case ikThinking, ikText, ikCallJSON, ikCallText:
			if s.state != 1 { // a content marker with no message open: a new model message
				s.closeMessage()
			}
			s.state, s.started = 2, false
			switch m {
			case ikThinking:
				s.route, s.reasoned = hmToReasoning, true
			case ikText:
				s.route = hmToContent
			default:
				s.route = hmToCall
				s.callBody.Reset()
			}
		case ikEnd:
			s.closeMessage()
		case ikEOS, ikUser, ikSystem, ikTool:
			s.closeMessage()
			s.done = true
		}
	}
	return out[hmToReasoning].String(), out[hmToContent].String()
}

// closeMessage ends the message being read: a call message keeps its name and body.
func (s *inklingSplitter) closeMessage() {
	if s.state == 2 && s.route == hmToCall {
		s.calls = append(s.calls, inklingCall{name: strings.TrimSpace(s.header), body: s.callBody.String()})
		s.callBody.Reset()
	}
	s.state, s.started, s.route = 0, false, hmToContent
}

func firstMarker(s string) (int, string) {
	best, bm := -1, ""
	for _, m := range ikMarkers {
		if i := strings.Index(s, m); i >= 0 && (best < 0 || i < best) {
			best, bm = i, m
		}
	}
	return best, bm
}

func (s *inklingSplitter) Flush() (reasoning, content string) {
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
	case s.state == 2 && s.route == hmToReasoning:
		r = rest
	case s.state == 2 && s.route == hmToCall:
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

// inklingToolCalls turns the calls a reply made into ToolCalls: the body's "args" object (or the body itself when it has none)
// compacted, under the message's name or the body's "name". A call whose arguments are not a JSON object is left out.
func inklingToolCalls(calls []inklingCall) []ToolCall {
	var out []ToolCall
	for _, c := range calls {
		var body struct {
			Name string          `json:"name"`
			Args json.RawMessage `json:"args"`
		}
		raw := bytes.TrimSpace([]byte(c.body))
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
		name := c.name
		if name == "" {
			name = body.Name
		}
		if name == "" {
			continue
		}
		out = append(out, ToolCall{Name: name, Arguments: json.RawMessage(compact.Bytes())})
	}
	return out
}

// parseInklingTools is ParseToolCalls for Inkling 2: the calls, and the answer text.
func parseInklingTools(out string) ([]ToolCall, string) {
	sp := &inklingSplitter{}
	_, c1 := sp.Push(out)
	_, c2 := sp.Flush()
	calls := inklingToolCalls(sp.calls)
	if len(calls) == 0 {
		return nil, out
	}
	return calls, strings.TrimSpace(c1 + c2)
}

package gemma4tools

import (
	"encoding/json"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

// Quote is Gemma 4's string-quote control marker inside tool declarations, calls, and responses.
const Quote = `<|"|>`

// MaxSchemaDepth bounds recursive JSON Schema traversal when rendering Gemma 4 tool declarations.
const MaxSchemaDepth = 64

var (
	toolMarks = []string{
		"<bos>", "<|turn>", "<turn|>", "<|think|>",
		"<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
		"<|tool_response>", "<tool_response|>", Quote,
		"<|channel>", "<channel|>",
	}
	quoteMarks = []string{Quote}
)

// Safe removes Gemma 4 control-token strings from an embedded schema or argument value in a single O(N) stack pass.
func Safe(s string) string {
	if !strings.ContainsAny(s, "<>") {
		return s
	}
	out := make([]byte, 0, len(s))
	for i := range len(s) {
		out = append(out, s[i])
		for len(out) > 0 && out[len(out)-1] == '>' {
			trimmed := false
			for _, m := range toolMarks {
				if len(out) >= len(m) && string(out[len(out)-len(m):]) == m {
					out = out[:len(out)-len(m)]
					trimmed = true
					break
				}
			}
			if !trimmed {
				break
			}
		}
	}
	return string(out)
}

// Opts configures generic Gemma 4 tool prompt rendering.
type Opts struct {
	Think    bool
	Hist     bool
	Old      bool
	Scaffold bool
}

// RenderTools renders system + turns + tool declarations in Gemma 4's generic segment format.
func RenderTools(has func(string) bool, system string, turns []chatcore.Turn, tools []chatcore.Tool, o Opts) []chatcore.Segment {
	b := chatcore.NewToolBuf(has, toolMarks...)
	b.Lit("<bos><|turn>system\n")
	if o.Think {
		b.Lit("<|think|>\n")
	}
	if s := strings.TrimSpace(system); s != "" {
		b.Txt(s)
	}
	for _, tl := range tools {
		b.Lit("<|tool>declaration:")
		b.Txt(tl.Name)
		b.Lit("{description:" + Quote)
		b.Txt(tl.Description)
		b.Lit(Quote + ",parameters:")
		b.LitMarks(Schema(tl.Parameters), quoteMarks)
		b.Lit("}<tool|>")
	}
	b.Lit("<turn|>\n")
	openModelTurn := false
	closeModelTurn := func() {
		if openModelTurn {
			b.Lit("<turn|>\n")
			openModelTurn = false
		}
	}
	lu := chatcore.LastUserIndex(turns)
	callName := map[string]string{}
	for i, m := range turns {
		switch m.Role {
		case "assistant":
			for _, c := range m.ToolCalls {
				if c.ID != "" {
					callName[c.ID] = c.Name
				}
			}
			content := m.Content
			channel := false
			if o.Hist {
				content = chatcore.StripChannels(content)
				channel = m.Reasoning != "" && i > lu && (!o.Old || len(m.ToolCalls) > 0)
			}
			if !o.Hist {
				closeModelTurn()
			}
			if !openModelTurn {
				b.Lit("<|turn>model\n")
			}
			if channel {
				b.Lit("<|channel>thought\n")
				b.Txt(m.Reasoning)
				b.Lit("\n<channel|>")
			}
			b.Txt(content)
			for _, c := range m.ToolCalls {
				b.Lit("<|tool_call>call:")
				b.Txt(c.Name)
				b.Lit("{")
				b.LitMarks(Args(c.Arguments), quoteMarks)
				b.Lit("}<tool_call|>")
			}
			openModelTurn = true
		case "tool":
			name := m.ToolName
			if n, ok := callName[m.ToolCallID]; ok && m.ToolCallID != "" && (o.Hist || name == "") {
				name = n
			}
			if o.Hist && name == "" {
				name = "unknown"
			}
			b.Lit("<|tool_response>response:")
			b.Txt(name)
			b.Lit("{value:" + Quote)
			b.Txt(m.Content)
			b.Lit(Quote + "}<tool_response|>")
		default:
			closeModelTurn()
			b.Lit("<|turn>user\n")
			b.Txt(m.Content)
			b.Lit("<turn|>\n")
		}
	}
	if !openModelTurn {
		b.Lit("<|turn>model\n")
		if !o.Think && o.Scaffold {
			b.Lit("<|channel>thought\n<channel|>")
		}
	} else if o.Old && len(turns) > 0 && turns[len(turns)-1].Role == "assistant" && len(turns[len(turns)-1].ToolCalls) == 0 {
		b.Lit("<turn|>\n<|turn>model\n")
	} else if o.Think && o.Hist && !o.Old && len(turns) > 0 && turns[len(turns)-1].Role == "tool" {
		b.Lit("<|channel>thought\n")
	}
	return b.Out()
}

// NativeOpts configures canonical Gemma 4 native tool prompt rendering.
type NativeOpts struct {
	Think              bool
	Scaffold           bool
	ContentBeforeCalls bool
}

// RenderNativeTools renders system + turns + tool declarations matching Gemma 4's canonical chat template.
func RenderNativeTools(has func(string) bool, system string, turns []chatcore.Turn, tools []chatcore.Tool, o NativeOpts) []chatcore.Segment {
	b := chatcore.NewToolBuf(has, toolMarks...)
	b.Lit("<bos>")
	sys := strings.TrimSpace(system)
	if o.Think || len(tools) > 0 || sys != "" {
		b.Lit("<|turn>system\n")
		if o.Think {
			b.Lit("<|think|>\n")
		}
		b.Txt(sys)
		for _, tl := range tools {
			b.Lit("<|tool>declaration:")
			b.Txt(tl.Name)
			b.Lit("{description:" + Quote)
			b.Txt(tl.Description)
			b.Lit(Quote + ",parameters:")
			b.LitMarks(Schema(tl.Parameters), quoteMarks)
			b.Lit("}<tool|>")
		}
		b.Lit("<turn|>\n")
	}
	prevMessageType := ""
	switch {
	case len(tools) > 0:
		prevMessageType = "tool"
	case o.Think:
		prevMessageType = "think"
	}
	prevNonToolRole := ""
	lu := chatcore.LastUserIndex(turns)
	for i, m := range turns {
		if m.Role == "tool" {
			continue
		}
		prevMessageType = ""
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		if !(role == "model" && prevNonToolRole == "assistant") {
			b.Lit("<|turn>")
			b.Txt(role)
			b.Lit("\n")
		}
		if m.Reasoning != "" && i > lu {
			b.Lit("<|channel>thought\n")
			b.Txt(m.Reasoning)
			b.Lit("\n<channel|>")
		}
		var content string
		if role == "model" {
			content = chatcore.StripChannels(m.Content)
		} else {
			content = strings.TrimSpace(m.Content)
		}
		contentFirst := o.ContentBeforeCalls && role == "model" && len(m.ToolCalls) > 0
		if contentFirst {
			b.Txt(content)
		}
		for _, c := range m.ToolCalls {
			b.Lit("<|tool_call>call:")
			b.Txt(c.Name)
			b.Lit("{")
			b.LitMarks(Args(c.Arguments), quoteMarks)
			b.Lit("}<tool_call|>")
			prevMessageType = "tool_call"
		}
		responded := false
		if len(m.ToolCalls) > 0 {
			for k := i + 1; k < len(turns) && turns[k].Role == "tool"; k++ {
				name := turns[k].ToolName
				if name == "" {
					name = "unknown"
				}
				for _, c := range m.ToolCalls {
					if c.ID != "" && c.ID == turns[k].ToolCallID {
						name = c.Name
					}
				}
				b.Lit("<|tool_response>response:")
				b.Txt(name)
				b.Lit("{value:" + Quote)
				b.Txt(turns[k].Content)
				b.Lit(Quote + "}<tool_response|>")
				responded = true
				prevMessageType = "tool_response"
			}
		}
		if !contentFirst {
			b.Txt(content)
		}
		hasContent := strings.TrimSpace(content) != ""

		nextRole, nextFound := "", false
		for j := i + 1; j < len(turns); j++ {
			if turns[j].Role != "tool" {
				nextRole, nextFound = turns[j].Role, true
				break
			}
		}
		continuesIntoNext := role == "model" && nextRole == "assistant" && (len(m.ToolCalls) == 0 || responded)
		switch {
		case prevMessageType == "tool_call" && !responded:
			b.Lit("<|tool_response>")
		case continuesIntoNext:
		case !(responded && !hasContent && !nextFound):
			b.Lit("<turn|>\n")
		}
		prevNonToolRole = m.Role
	}
	if prevMessageType != "tool_response" && prevMessageType != "tool_call" {
		b.Lit("<|turn>model\n")
		if !o.Think && o.Scaffold {
			b.Lit("<|channel>thought\n<channel|>")
		}
	} else if prevMessageType == "tool_response" && o.Think {
		b.Lit("<|channel>thought\n")
	}
	return b.Out()
}

// ParseTools extracts Gemma 4 tool calls and leading text from a model reply.
func ParseTools(out string) ([]chatcore.ToolCall, string) {
	lead := out
	if before, _, ok := strings.Cut(out, "<|tool_call>"); ok {
		lead = before
	}
	var calls []chatcore.ToolCall
	for rest := out; ; {
		_, after, ok := strings.Cut(rest, "<|tool_call>call:")
		if !ok {
			break
		}
		namePart, bodyRest, ok := strings.Cut(after, "{")
		if !ok {
			break
		}
		name := strings.TrimSpace(namePart)
		body, next, hasEnd := strings.Cut(bodyRest, "}<tool_call|>")
		rest = next
		calls = append(calls, chatcore.ToolCall{Name: name, Arguments: BodyToJSON(body)})
		if !hasEnd {
			break
		}
	}
	return calls, strings.TrimSpace(lead)
}

// Schema renders a JSON Schema in Gemma 4's parameter syntax (keys sorted, types upper-cased, strings in Quote markers).
func Schema(raw json.RawMessage) string {
	var s map[string]any
	if json.Unmarshal(raw, &s) != nil {
		return "{type:" + Quote + "OBJECT" + Quote + "}"
	}
	return schemaMap(s, 0)
}

func schemaVal(v any, depth int) string {
	if v == nil {
		return "{}"
	}
	if m, ok := v.(map[string]any); ok {
		return schemaMap(m, depth)
	}
	return "{type:" + Quote + "OBJECT" + Quote + "}"
}

func schemaMap(s map[string]any, depth int) string {
	if depth >= MaxSchemaDepth {
		return "{type:" + Quote + "OBJECT" + Quote + "}"
	}
	var parts []string
	for _, k := range slices.Sorted(maps.Keys(s)) {
		v := s[k]
		switch k {
		case "type":
			parts = append(parts, "type:"+Quote+Safe(strings.ToUpper(toStr(v)))+Quote)
		case "description":
			parts = append(parts, "description:"+Quote+Safe(toStr(v))+Quote)
		case "enum":
			parts = append(parts, "enum:"+strList(v))
		case "required":
			parts = append(parts, "required:"+strList(v))
		case "properties":
			parts = append(parts, "properties:"+props(v, depth+1))
		case "items":
			parts = append(parts, "items:"+schemaVal(v, depth+1))
		}
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func props(v any, depth int) string {
	m, ok := v.(map[string]any)
	if !ok || depth >= MaxSchemaDepth {
		return "{}"
	}
	var parts []string
	for _, k := range slices.Sorted(maps.Keys(m)) {
		parts = append(parts, Safe(k)+":"+schemaVal(m[k], depth+1))
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func strList(v any) string {
	arr, ok := v.([]any)
	if !ok {
		return "[]"
	}
	var parts []string
	for _, e := range arr {
		parts = append(parts, Quote+Safe(toStr(e))+Quote)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// Args renders a JSON arguments object as Gemma 4's key:value call body.
func Args(raw json.RawMessage) string {
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return ""
	}
	var parts []string
	for _, k := range slices.Sorted(maps.Keys(m)) {
		parts = append(parts, Safe(k)+":"+value(m[k]))
	}
	return strings.Join(parts, ",")
}

func value(v any) string {
	switch x := v.(type) {
	case string:
		return Quote + Safe(x) + Quote
	case bool:
		return strconv.FormatBool(x)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case map[string]any:
		keys := slices.Sorted(maps.Keys(x))
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, Safe(k)+":"+value(x[k]))
		}
		return "{" + strings.Join(parts, ",") + "}"
	case []any:
		parts := make([]string, 0, len(x))
		for _, e := range x {
			parts = append(parts, value(e))
		}
		return "[" + strings.Join(parts, ",") + "]"
	case nil:
		return "null"
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

// BodyToJSON parses a Gemma 4 "key:value,..." call body into a JSON object.
func BodyToJSON(body string) json.RawMessage {
	out := map[string]any{}
	for _, kv := range splitPairs(body) {
		before, after, ok := strings.Cut(kv, ":")
		if !ok {
			continue
		}
		key := strings.TrimSpace(before)
		val := strings.TrimSpace(after)
		out[key] = parseValue(val)
	}
	b, _ := json.Marshal(out)
	return b
}

func parseValue(val string) any {
	switch {
	case strings.HasPrefix(val, Quote):
		return strings.TrimSuffix(strings.TrimPrefix(val, Quote), Quote)
	case val == "true" || val == "false":
		return val == "true"
	case val == "null":
		return nil
	case strings.HasPrefix(val, "{") && strings.HasSuffix(val, "}"):
		inner := val[1 : len(val)-1]
		m := map[string]any{}
		for _, kv := range splitPairs(inner) {
			k, v, ok := strings.Cut(kv, ":")
			if !ok {
				continue
			}
			m[strings.TrimSpace(k)] = parseValue(strings.TrimSpace(v))
		}
		return m
	case strings.HasPrefix(val, "[") && strings.HasSuffix(val, "]"):
		inner := strings.TrimSpace(val[1 : len(val)-1])
		if inner == "" {
			return []any{}
		}
		var arr []any
		for _, e := range splitPairs(inner) {
			arr = append(arr, parseValue(strings.TrimSpace(e)))
		}
		return arr
	}
	if f, err := strconv.ParseFloat(val, 64); err == nil {
		return f
	}
	return val
}

func splitPairs(body string) []string {
	var pairs []string
	inStr := false
	nest := 0
	start := 0
	for i := 0; i < len(body); i++ {
		if strings.HasPrefix(body[i:], Quote) {
			inStr = !inStr
			i += len(Quote) - 1
			continue
		}
		if inStr {
			continue
		}
		switch body[i] {
		case '{', '[':
			nest++
		case '}', ']':
			if nest > 0 {
				nest--
			}
		case ',':
			if nest == 0 {
				pairs = append(pairs, body[start:i])
				start = i + 1
			}
		}
	}
	if start < len(body) {
		pairs = append(pairs, body[start:])
	}
	return pairs
}

func toStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	b, _ := json.Marshal(v)
	return string(b)
}

// DetectContentBeforeCalls reports whether a Gemma 4 chat template writes an assistant turn's text before its tool calls.
func DetectContentBeforeCalls(tmpl string) bool {
	flat := strings.Join(strings.Fields(tmpl), "")
	before := strings.Index(flat, "{%-ifrole=='model'andmessage.get('tool_calls')-%}{{-captured_content-}}{%-endif-%}")
	call := strings.Index(flat, "'<|tool_call>call:'")
	return before >= 0 && call > before && strings.Contains(flat, "{%-ifnot(role=='model'andmessage.get('tool_calls'))-%}{{-captured_content-}}")
}

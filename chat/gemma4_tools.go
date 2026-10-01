package chat

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// Gemma 4 tool calling — a bespoke non-JSON micro-language, distinct from the
// other families. Tools are declared as
//
//	<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:<gschema>}<tool|>
//
// a call is
//
//	<|tool_call>call:NAME{key:<|"|>val<|"|>,…}<tool_call|>
//
// and a result is <|tool_response>response:NAME{value:<|"|>…<|"|>}<tool_response|>.
// Strings are wrapped in the <|"|> quote marker; types are upper-cased; schema
// keys are emitted alphabetically.

const gq = `<|"|>` // Gemma's string-quote marker

func renderGemma4Tools(system string, turns []Turn, tools []Tool, think, hist, old bool) string {
	var b strings.Builder
	b.WriteString("<bos><|turn>system\n")
	if think {
		b.WriteString("<|think|>\n")
	}
	// No separator between the system text and the first declaration: upstream's template renders
	// "<|turn>system\n{system}<|tool>declaration:…" (measured against HF with a system prompt, 2026-09-30 —
	// the golden's only cases had none, so the "\n" that used to be written here was never compared).
	if s := strings.TrimSpace(system); s != "" {
		b.WriteString(s)
	}
	for _, tl := range tools {
		b.WriteString("<|tool>declaration:" + tl.Name + "{description:" + gq + tl.Description + gq +
			",parameters:" + gemmaSchema(tl.Parameters) + "}<tool|>")
	}
	b.WriteString("<turn|>\n")
	// M-20: THE TOOL RESPONSES SIT INSIDE THE MODEL TURN THAT MADE THE CALL. goinfer used to
	// close the turn with "<turn|>\n" after the calls and then re-open a
	// "<|turn>model\n<|channel>thought\n<channel|>" scaffold after the response; the upstream
	// template does neither. Both are visible in testdata/chat_goldens/tools_gemma4.json's
	// `call_result` case, which was committed and never read by any test.
	//
	// So the turn is closed lazily: an assistant turn whose calls are answered by tool turns
	// stays open until something that is not a tool response follows it.
	openModelTurn := false
	closeModelTurn := func() {
		if openModelTurn {
			b.WriteString("<turn|>\n")
			openModelTurn = false
		}
	}
	lu := lastUserIndex(turns)
	callName := map[string]string{} // call id -> function name, so a result that carries only the id still names its function
	for i, m := range turns {
		switch m.Role {
		case "assistant":
			for _, c := range m.ToolCalls {
				if c.ID != "" {
					callName[c.ID] = c.Name
				}
			}
			// The tool loop in progress keeps each turn's reasoning as a thought channel at the top of the turn (history.go).
			// A model turn that CONTINUES an open one (calls answered by tool turns) carries no new turn marker, but its
			// reasoning channel is still written.
			content := m.Content
			var channel string
			if hist {
				content = stripChannels(content)
				// The earlier template keeps the reasoning only of a turn that makes a call (`thinking_text and … and message.get('tool_calls')`).
				if m.Reasoning != "" && i > lu && (!old || len(m.ToolCalls) > 0) {
					channel = "<|channel>thought\n" + m.Reasoning + "\n<channel|>"
				}
			}
			// The managed rendering continues an open model turn the way the template does (an assistant turn after tool
			// responses has no turn marker of its own); the generic rendering keeps closing it and opening a new one, byte for
			// byte as before the history rule existed.
			if !hist {
				closeModelTurn()
			}
			if openModelTurn {
				b.WriteString(channel + content)
			} else {
				b.WriteString("<|turn>model\n" + channel + content)
			}
			for _, c := range m.ToolCalls {
				b.WriteString("<|tool_call>call:" + c.Name + "{" + gemmaArgs(c.Arguments) + "}<tool_call|>")
			}
			openModelTurn = true
		case "tool":
			// Deliberately does NOT close or re-open a turn: it continues the model turn above.
			// The function a result answers is named by its call id when it carries one (the template's own rule — OpenAI clients
			// often send only `tool_call_id`; `name` is optional there), else by the name the result gave. A result with neither is
			// "unknown" in the template; the generic rendering keeps writing an empty name, byte for byte as before.
			name := m.ToolName
			if n, ok := callName[m.ToolCallID]; ok && m.ToolCallID != "" && (hist || name == "") {
				name = n
			}
			if hist && name == "" {
				name = "unknown"
			}
			b.WriteString("<|tool_response>response:" + name + "{value:" + gq + m.Content + gq + "}<tool_response|>")
		default:
			closeModelTurn()
			b.WriteString("<|turn>user\n" + m.Content + "<turn|>\n")
		}
	}
	// The generation prompt follows a USER turn, not a tool response — upstream's
	// add_generation_prompt emits nothing after a tool_response, because the model is already
	// mid-turn and simply continues.
	if !openModelTurn {
		b.WriteString("<|turn>model\n")
		if !think && !old { // thinking on leaves the channel for the model to open; the earlier template has no scaffold at all
			b.WriteString("<|channel>thought\n<channel|>")
		}
	} else if old && len(turns) > 0 && turns[len(turns)-1].Role == "assistant" && len(turns[len(turns)-1].ToolCalls) == 0 {
		// The earlier template closes a call-less assistant message even when it follows tool results, then writes the generation prompt.
		b.WriteString("<turn|>\n<|turn>model\n")
	} else if think && hist && !old && len(turns) > 0 && turns[len(turns)-1].Role == "tool" { // the earlier template writes nothing here
		// The model is still in the turn that made the call. With thinking on the template reopens its thought channel for
		// the next step, so the reply starts inside it (Template.PromptOpensThinkFor).
		b.WriteString("<|channel>thought\n")
	}
	return b.String()
}

func parseGemma4Tools(out string) ([]ToolCall, string) {
	lead := out
	if before, _, ok := strings.Cut(out, "<|tool_call>"); ok {
		lead = before
	}
	var calls []ToolCall
	for rest := out; ; {
		i := strings.Index(rest, "<|tool_call>call:")
		if i < 0 {
			break
		}
		rest = rest[i+len("<|tool_call>call:"):]
		brace := strings.IndexByte(rest, '{')
		if brace < 0 {
			break
		}
		name := strings.TrimSpace(rest[:brace])
		rest = rest[brace+1:]
		end := strings.Index(rest, "}<tool_call|>")
		body := rest
		if end >= 0 {
			body = rest[:end]
			rest = rest[end+len("}<tool_call|>"):]
		}
		calls = append(calls, ToolCall{Name: name, Arguments: gemmaBodyToJSON(body)})
		if end < 0 {
			break
		}
	}
	return calls, strings.TrimSpace(lead)
}

// gemmaSchema renders a JSON Schema in Gemma's parameter syntax (keys alphabetical,
// types upper-cased, strings in <|"|> quotes).
func gemmaSchema(raw json.RawMessage) string {
	var s map[string]any
	if json.Unmarshal(raw, &s) != nil {
		return "{type:" + gq + "OBJECT" + gq + "}"
	}
	keys := make([]string, 0, len(s))
	for k := range s {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		v := s[k]
		switch k {
		case "type":
			parts = append(parts, "type:"+gq+strings.ToUpper(toStr(v))+gq)
		case "description":
			parts = append(parts, "description:"+gq+toStr(v)+gq)
		case "enum":
			parts = append(parts, "enum:"+gemmaStrList(v))
		case "required":
			parts = append(parts, "required:"+gemmaStrList(v))
		case "properties":
			parts = append(parts, "properties:"+gemmaProps(v))
		case "items":
			if sub, _ := json.Marshal(v); sub != nil {
				parts = append(parts, "items:"+gemmaSchema(sub))
			}
		}
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func gemmaProps(v any) string {
	m, ok := v.(map[string]any)
	if !ok {
		return "{}"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		sub, _ := json.Marshal(m[k])
		parts = append(parts, k+":"+gemmaSchema(sub))
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func gemmaStrList(v any) string {
	arr, ok := v.([]any)
	if !ok {
		return "[]"
	}
	var parts []string
	for _, e := range arr {
		parts = append(parts, gq+toStr(e)+gq)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// gemmaArgs renders a JSON arguments object as Gemma's key:value body.
func gemmaArgs(raw json.RawMessage) string {
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		parts = append(parts, k+":"+gemmaValue(m[k]))
	}
	return strings.Join(parts, ",")
}

// gemmaValue renders one argument value in Gemma's micro-language.
//
// M-20: the default arm used to json.Marshal, so an object-typed parameter came out as
// {"limit":5} — JSON, not Gemma syntax — inside a body the model reads as Gemma syntax. Upstream
// renders a mapping as {k:v,…} with the same quoting as the top level, which is what the
// declaration half of this file already does for schemas (gemmaSchema). Any tool with an
// object-typed or array-typed parameter hit this.
func gemmaValue(v any) string {
	switch x := v.(type) {
	case string:
		return gq + x + gq
	case bool:
		return strconv.FormatBool(x)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys) // deterministic, as gemmaArgs does
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, k+":"+gemmaValue(x[k]))
		}
		return "{" + strings.Join(parts, ",") + "}"
	case []any:
		parts := make([]string, 0, len(x))
		for _, e := range x {
			parts = append(parts, gemmaValue(e))
		}
		return "[" + strings.Join(parts, ",") + "]"
	case nil:
		return "null"
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

// gemmaBodyToJSON parses a "key:value,…" call body into a JSON object. String
// values are <|"|>-quoted; bare values are parsed as numbers/bools (else string).
func gemmaBodyToJSON(body string) json.RawMessage {
	out := map[string]any{}
	for _, kv := range splitGemmaPairs(body) {
		before, after, ok := strings.Cut(kv, ":")
		if !ok {
			continue
		}
		key := strings.TrimSpace(before)
		val := strings.TrimSpace(after)
		out[key] = gemmaParseValue(val)
	}
	b, _ := json.Marshal(out)
	return b
}

// gemmaParseValue is gemmaValue's inverse for one rendered value: <|"|>-quoted → string,
// true/false → bool, {…} → object, […] → array, else a number, else the bare text.
//
// M-20: nested values had no case at all, so an object argument came back as two broken string
// keys. Recursive here for the same reason gemmaValue is recursive — a renderer and a parser
// that disagree about nesting produce arguments the tool silently mis-receives.
func gemmaParseValue(val string) any {
	switch {
	case strings.HasPrefix(val, gq):
		return strings.TrimSuffix(strings.TrimPrefix(val, gq), gq)
	case val == "true" || val == "false":
		return val == "true"
	case val == "null":
		return nil
	case strings.HasPrefix(val, "{") && strings.HasSuffix(val, "}"):
		inner := val[1 : len(val)-1]
		m := map[string]any{}
		for _, kv := range splitGemmaPairs(inner) {
			k, v, ok := strings.Cut(kv, ":")
			if !ok {
				continue
			}
			m[strings.TrimSpace(k)] = gemmaParseValue(strings.TrimSpace(v))
		}
		return m
	case strings.HasPrefix(val, "[") && strings.HasSuffix(val, "]"):
		inner := strings.TrimSpace(val[1 : len(val)-1])
		if inner == "" {
			return []any{}
		}
		var arr []any
		for _, e := range splitGemmaPairs(inner) {
			arr = append(arr, gemmaParseValue(strings.TrimSpace(e)))
		}
		return arr
	}
	if f, err := strconv.ParseFloat(val, 64); err == nil {
		return f
	}
	return val
}

// splitGemmaPairs splits on commas that are neither inside a <|"|>…<|"|> string nor inside a
// nested {…} / […].
//
// M-20: it used to track quoting only, so opts:{limit:5,sort:<|"|>asc<|"|>} split at the INNER
// comma and parsed to {"opts":"{limit:5","sort":"asc}"} — two keys, both wrong, no error.
func splitGemmaPairs(body string) []string {
	var pairs []string
	inStr := false
	nest := 0
	start := 0
	for i := 0; i < len(body); i++ {
		if strings.HasPrefix(body[i:], gq) {
			inStr = !inStr
			i += len(gq) - 1
			continue
		}
		if inStr {
			continue // braces and commas inside a string are literal text
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

// renderGemma4NativeTools renders system + turns + tool declarations the way Gemma 4's CANONICAL chat template does — a port of its
// message loop, not a variation on renderGemma4Tools — and is what `-tool-format template` selects for such a checkpoint. The variable
// names are the template's (prev_message_type, prev_non_tool_role, continues_into_next) so the two can be read side by side. Three
// things differ from goinfer's own rendering, all of them the template's:
//
//   - an assistant message's TEXT is written after its calls and their results, not before the calls;
//   - the model turn is CLOSED (`<turn|>`) right after that text, when the message has results and text;
//   - the generation prompt writes no turn header after a tool result, whether or not the turn was closed.
//
// The last two make a prompt that ends `…<tool_response|>text<turn|>\n` with nothing after it when an agent client replays a turn that
// had a preamble. Whether that is better or worse for the model than goinfer's own order is a question for a measurement, not for this
// file: the format is opt-in.
func renderGemma4NativeTools(system string, turns []Turn, tools []Tool, think bool) string {
	var b strings.Builder
	b.WriteString("<bos>")
	sys := strings.TrimSpace(system)
	if think || len(tools) > 0 || sys != "" {
		b.WriteString("<|turn>system\n")
		if think {
			b.WriteString("<|think|>\n")
		}
		b.WriteString(sys)
		for _, tl := range tools {
			b.WriteString("<|tool>declaration:" + tl.Name + "{description:" + gq + tl.Description + gq +
				",parameters:" + gemmaSchema(tl.Parameters) + "}<tool|>")
		}
		b.WriteString("<turn|>\n")
	}
	prevMessageType := ""
	switch {
	case len(tools) > 0:
		prevMessageType = "tool"
	case think:
		prevMessageType = "think"
	}
	prevNonToolRole := ""
	lu := lastUserIndex(turns)
	for i, m := range turns {
		if m.Role == "tool" {
			continue // written by the assistant message that made the call (the forward scan below)
		}
		prevMessageType = ""
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		if !(role == "model" && prevNonToolRole == "assistant") { // continue_same_model_turn
			b.WriteString("<|turn>" + role + "\n")
		}
		if m.Reasoning != "" && i > lu { // thinking_gate (preserve_thinking is off)
			b.WriteString("<|channel>thought\n" + m.Reasoning + "\n<channel|>")
		}
		for _, c := range m.ToolCalls {
			b.WriteString("<|tool_call>call:" + c.Name + "{" + gemmaArgs(c.Arguments) + "}<tool_call|>")
			prevMessageType = "tool_call"
		}
		responded := false // ns_tr_out.flag
		if len(m.ToolCalls) > 0 {
			for k := i + 1; k < len(turns) && turns[k].Role == "tool"; k++ {
				name := turns[k].ToolName
				if name == "" {
					name = "unknown"
				}
				for _, c := range m.ToolCalls { // resolve tool_call_id to the function name
					if c.ID != "" && c.ID == turns[k].ToolCallID {
						name = c.Name
					}
				}
				b.WriteString("<|tool_response>response:" + name + "{value:" + gq + turns[k].Content + gq + "}<tool_response|>")
				responded = true
				prevMessageType = "tool_response"
			}
		}
		var content string // captured_content
		if role == "model" {
			content = stripChannels(m.Content)
		} else {
			content = strings.TrimSpace(m.Content)
		}
		b.WriteString(content)
		hasContent := strings.TrimSpace(content) != ""

		nextRole, nextFound := "", false // the next non-tool message
		for j := i + 1; j < len(turns); j++ {
			if turns[j].Role != "tool" {
				nextRole, nextFound = turns[j].Role, true
				break
			}
		}
		// (The template's `and (not tool_calls or tool_responses)` is kept as written; with the switch below it is implied, because a call
		// with no result takes the first case before this one is consulted.)
		continuesIntoNext := role == "model" && nextRole == "assistant" && (len(m.ToolCalls) == 0 || responded)
		switch {
		case prevMessageType == "tool_call" && !responded:
			b.WriteString("<|tool_response>")
		case continuesIntoNext:
		case !(responded && !hasContent && !nextFound):
			b.WriteString("<turn|>\n")
		}
		prevNonToolRole = m.Role
	}
	if prevMessageType != "tool_response" && prevMessageType != "tool_call" {
		b.WriteString("<|turn>model\n")
		if !think {
			b.WriteString("<|channel>thought\n<channel|>")
		}
	} else if prevMessageType == "tool_response" && think {
		b.WriteString("<|channel>thought\n")
	}
	return b.String()
}

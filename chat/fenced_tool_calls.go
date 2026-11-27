package chat

// WithLenientToolCalls returns a copy of the template that also accepts one schema-validated
// JSON tool call inside a trailing code fence on families accepting bare calls.
func (t *Template) WithLenientToolCalls(on bool) *Template {
	if t == nil || t.lenientFenced == on {
		return t
	}
	c := *t
	c.lenientFenced = on
	return &c
}

// LenientToolCalls reports whether WithLenientToolCalls is active on t.
func (t *Template) LenientToolCalls() bool {
	return t != nil && t.lenientFenced && t.AcceptsBareToolCall()
}

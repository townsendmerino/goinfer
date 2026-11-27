package chat

import (
	"strconv"

	"github.com/townsendmerino/goinfer/internal/chat/inkling2"
)

const (
	ikUser  = inkling2.User
	ikModel = inkling2.Model
	ikText  = inkling2.Text
	ikEnd   = inkling2.End
	ikEOS   = inkling2.EOS
)

var inklingEfforts = inkling2.Efforts

// Inkling2 returns Inkling 2's chat template with its default reasoning effort (0.9).
func Inkling2() *Template { return inkling2Template(-1) }

func inkling2Template(effort float64) *Template {
	t := newTemplate("inkling2", []string{ikEOS}, nil)
	e := ""
	if effort >= 0 {
		e = strconv.FormatFloat(effort, 'g', -1, 64)
	}
	return t.inklingWith(e, ThinkAsIs)
}

func (t *Template) inklingWith(effort string, m ThinkMode) *Template {
	c := *t
	c.effort, c.think = effort, m
	cp := &c
	cp.render = func(system string, turns []Turn) []Segment { return cp.inkling2Segments(system, turns, nil) }
	return cp
}

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
	return inkling2.Segments(t.inklingEffort(), system, toCoreTurns(turns), toCoreTools(tools))
}

func parseInklingTools(out string) ([]ToolCall, string) {
	calls, lead := inkling2.ParseTools(out)
	return fromCoreCalls(calls), lead
}

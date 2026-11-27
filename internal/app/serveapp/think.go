package serveapp

import (
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
	"github.com/townsendmerino/goinfer/internal/codec"
)

type (
	reasoningFormat = streamframe.ReasoningFormat
	thinkSettings   = streamframe.ThinkSettings
	thinkRequest    = streamframe.ThinkRequest
	thinkOut        = streamframe.ThinkOut
)

const (
	rfSplit  = streamframe.FormatSplit
	rfLegacy = streamframe.FormatLegacy
	rfNone   = streamframe.FormatNone
)

var (
	parseReasoningFormat = streamframe.ParseReasoningFormat
	newThinkOut          = streamframe.NewThinkOut
)

func (c *config) thinkDefault() chat.ThinkMode {
	if strings.TrimSpace(c.thinking) == "" {
		return chat.ThinkTemplate
	}
	m, _ := chat.ParseThinkMode(c.thinking)
	return m
}

func (c *config) tuneTemplate(t *chat.Template) *chat.Template {
	return t.WithThinking(c.thinkDefault()).WithToolFormat(c.toolFormatDefault()).WithLenientToolCalls(c.lenientToolCalls)
}

func (c *config) toolFormatDefault() chat.ToolFormat {
	f, _ := chat.ParseToolFormat(c.toolFormat)
	return f
}

func (c *config) reasoningFormat() reasoningFormat {
	f, _ := parseReasoningFormat(c.reasoningFmt)
	return f
}

func (s *server) resolveThink(tr thinkRequest) (thinkSettings, error) {
	return streamframe.ResolveThink(s.cfg.thinkDefault(), s.cfg.reasoningFormat(), tr)
}

func (lm *loadedModel) templateFor(ts thinkSettings) *chat.Template {
	t := lm.tmpl.WithReasoningEffort(ts.Effort)
	if t == nil || !ts.Explicit {
		return t
	}
	return t.WithThinking(ts.Mode)
}

func (lm *loadedModel) constrainedTemplate(ts thinkSettings) *chat.Template {
	t := lm.templateFor(ts)
	if t.PromptOpensThink() {
		return t.WithThinking(chat.ThinkOff)
	}
	return t
}

func (lm *loadedModel) promptForT(tmpl *chat.Template, system string, turns []chat.Turn) ([]int, error) {
	if tmpl != nil {
		return codec.EncodePrompt(lm.tk, tmpl, system, turns, nil)
	}
	return lm.encode(rawPrompt(system, turns))
}

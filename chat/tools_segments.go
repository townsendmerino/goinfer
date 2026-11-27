package chat

import "github.com/townsendmerino/goinfer/internal/chat/chatcore"

type toolBuf = chatcore.ToolBuf

func (t *Template) newToolBuf(marks ...string) *toolBuf {
	return chatcore.NewToolBuf(t.has, marks...)
}

package chat

import "github.com/townsendmerino/goinfer/internal/chat/chatcore"

var chatMLDefaultSystem = chatcore.ChatMLDefaultSystem

// systemOr is the system message to render: the caller's, or the template's default when the caller sent none.
func (t *Template) systemOr(system string) string {
	if system == "" {
		return t.defaultSystem
	}
	return system
}

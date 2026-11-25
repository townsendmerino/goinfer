package serveapp

import "github.com/townsendmerino/goinfer/internal/app/serveapp/sessions"

// sessionLRU aliases sessions.LRU, the prefix-keyed KV session cache, cold-disk tier, and snapshot persistence.
type sessionLRU = sessions.LRU

var newSessionLRU = sessions.New

func sessionDirOK(dir string) error { return sessions.DirOK(dir) }

func pickSessionFor(lists [][]int, firsts []int, prompt []int, spare bool) int {
	return sessions.PickFor(lists, firsts, prompt, spare)
}

//go:build goinfer_staticcheck_canary

package canary

import "strings"

type holder struct {
	deadField int
	b         strings.Builder
}

func (h *holder) String() string { return h.b.String() }

// V keeps holder reachable, so only the field is unused.
var V holder

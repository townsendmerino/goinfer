package streamframe

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
)

// BudgetFlag represents a parsed -reasoning-budget server flag value.
type BudgetFlag struct {
	Unlimited bool
	N         int // positive token cap; 0 with !Unlimited means auto
}

// ParseBudgetFlag parses "", "auto", "unlimited", or a positive integer token ceiling.
func ParseBudgetFlag(s string) (BudgetFlag, error) {
	switch v := strings.ToLower(strings.TrimSpace(s)); v {
	case "", "auto":
		return BudgetFlag{}, nil
	case "unlimited":
		return BudgetFlag{Unlimited: true}, nil
	default:
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return BudgetFlag{}, fmt.Errorf("want auto, unlimited, or a positive number of tokens")
		}
		return BudgetFlag{N: n}, nil
	}
}

// ResolveBudget computes the reasoning-token ceiling for a turn of length maxTokens given server default f and
// per-request explicit budget (0 when unset), clamping to chat.BudgetRoom so answer room remains.
func ResolveBudget(f BudgetFlag, explicit, maxTokens int) (limit int, ok bool) {
	room, ok := chat.BudgetRoom(maxTokens)
	if !ok {
		return 0, false
	}
	limit = room
	if explicit > 0 {
		limit = min(explicit, room)
	} else if f.N > 0 {
		limit = min(f.N, room)
	} else if f.Unlimited {
		return 0, false
	}
	return limit, limit >= 1
}

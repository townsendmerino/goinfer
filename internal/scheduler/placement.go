package scheduler

// Placement chooses among a model's resident sessions, given each one's tokens. A session a generation holds is
// passed as nil tokens (or marked busy): its tokens are being written and are never read here.

// CommonPrefix is the length of the longest common leading run of a and b.
func CommonPrefix(a, b []int) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// BestExtend is the index of the session whose common prefix with prompt is longest, among those whose match beats
// what that same session shares with every other one, or -1. A preamble shared by several sessions never qualifies by
// itself, so conversations that open the same way keep their own sessions; an exact continuation always qualifies, and
// so does a partial match (a stop-string tail, a mid-word cut, an edited last message).
func BestExtend(sessions [][]int, prompt []int) int {
	best, bestMatch := -1, 0
	for i, toks := range sessions {
		match := CommonPrefix(toks, prompt)
		if match == 0 {
			continue
		}
		floor := 0
		for j, other := range sessions {
			if j != i {
				floor = max(floor, CommonPrefix(toks, other))
			}
		}
		if match > floor && match > bestMatch {
			best, bestMatch = i, match
		}
	}
	return best
}

// PickSession is the session prompt extends, or -1 for a fresh one: BestExtend's candidate, unless there is room for a
// fresh session (spare) and the request would keep less of the candidate than it overwrites (the room rule).
// firsts[i] is the length of the first prompt of the conversation session i holds (0: unknown); a request that keeps
// all of it is that conversation rolling back, and reuses its session however much it gives up. With no room, reusing
// the candidate is the eviction, so it stands.
func PickSession(sessions [][]int, firsts []int, prompt []int, spare bool) int {
	i := BestExtend(sessions, prompt)
	if i >= 0 && spare {
		keep := CommonPrefix(sessions[i], prompt)
		if own := i < len(firsts) && KeepsFirst(firsts[i], keep); !own && keep < len(sessions[i])-keep {
			return -1
		}
	}
	return i
}

// KeepsFirst reports whether a request whose common prefix with a session is keep tokens keeps the whole first prompt
// of that session's conversation, first tokens long (0: unknown, never).
func KeepsFirst(first, keep int) bool { return first > 0 && keep >= first }

// ColdestIdle is the index of the least recently used session no generation holds, or -1 when every one is busy. busy
// is in recency order, most recent first.
func ColdestIdle(busy []bool) int {
	for i := len(busy) - 1; i >= 0; i-- {
		if !busy[i] {
			return i
		}
	}
	return -1
}

// PickDonor is the idle session that can give a new prompt the most rows by copy, and how many: hits[i] is session i's
// shared hit, in recency order. Among equal hits the least recently used wins, which is the session a fresh slot would
// evict, so taking it in place saves the copy. -1 when no idle session has a hit.
func PickDonor(hits []int, busy []bool) (donor, hit int) {
	donor = -1
	for i, h := range hits {
		if busy[i] {
			continue
		}
		if h > 0 && h >= hit {
			donor, hit = i, h
		}
	}
	return donor, hit
}

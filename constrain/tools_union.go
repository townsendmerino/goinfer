package constrain

import "fmt"

// ToolSpec is one candidate tool for ToolCallsGrammar: its name and its JSON-Schema for the
// arguments object (nil/empty = takes no arguments, as in ToolCallGrammar).
type ToolSpec struct {
	Name       string
	Parameters []byte
}

// ToolCallsGrammar constrains ONE tool call to any of several tools (task T1,
// docs/tasks/task-tool-grammar-union-2026-09.md). It is the union of the single-tool
// grammars ToolCallGrammar already builds, run as parallel branches: a byte is legal iff at
// least one live branch accepts it, and a branch that rejects a committed byte is dropped.
// Every branch shares the wrapper and the object shape; they differ only in the "name"
// const and in what follows it, so the set collapses to one branch as soon as the name
// discriminates — and, because each branch is the complete single-tool grammar, that holds
// whichever order the model writes the object's keys in.
//
// Provable by construction: the language is exactly the union of the single-tool languages,
// so every string it accepts is a well-formed call to exactly one supplied tool, with
// arguments matching that tool's schema, and no string naming an absent tool is accepted.
// schemaGrammar is not modified (the task's ground rule 3).
//
// Duplicate names collapse to their first spec. An empty tool list, or a tool whose schema
// cannot be compiled, is an error: the caller then decodes unconstrained, as today, rather
// than silently dropping a tool the model was offered (ground rule 1 — the model keeps its
// choice).
func ToolCallsGrammar(prefix, suffix, argsKey string, array bool, tools []ToolSpec) (Grammar, error) {
	if len(tools) == 0 {
		return nil, fmt.Errorf("constrain: no tools")
	}
	seen := make(map[string]bool, len(tools))
	var branches []Grammar
	for _, t := range tools {
		if seen[t.Name] {
			continue
		}
		seen[t.Name] = true
		g, err := ToolCallGrammar(prefix, suffix, argsKey, t.Name, array, t.Parameters)
		if err != nil {
			return nil, fmt.Errorf("constrain: tool %q: %w", t.Name, err)
		}
		branches = append(branches, g)
	}
	u := &unionGrammar{all: branches}
	u.Reset()
	return u, nil
}

// unionGrammar is the parallel-branch union behind ToolCallsGrammar. live holds the branches
// still consistent with every committed byte; all keeps the originals so Reset can restore
// the full set.
type unionGrammar struct {
	all  []Grammar
	live []Grammar
}

func (u *unionGrammar) Reset() {
	u.live = u.live[:0]
	for _, g := range u.all {
		g.Reset()
		u.live = append(u.live, g)
	}
}

// Clone copies every live branch; the clone's Reset restores its OWN copies of the full set,
// never the original's, so lookahead on a clone cannot disturb the live grammar. Clone is on
// the grammar-fused speculative path (drafter lookahead), which is why it must be exact.
func (u *unionGrammar) Clone() Grammar {
	c := &unionGrammar{all: make([]Grammar, len(u.all)), live: make([]Grammar, 0, len(u.live))}
	for i, g := range u.all {
		c.all[i] = g.Clone()
	}
	for _, g := range u.live {
		c.live = append(c.live, g.Clone())
	}
	return c
}

func (u *unionGrammar) TryBytes(bs []byte) bool {
	for _, g := range u.live {
		if g.TryBytes(bs) {
			return true
		}
	}
	return false
}

func (u *unionGrammar) Commit(bs []byte) {
	kept := u.live[:0]
	for _, g := range u.live {
		if g.TryBytes(bs) {
			g.Commit(bs)
			kept = append(kept, g)
		}
	}
	// kept aliases live's backing array; zero the tail so dropped branches can be collected.
	for i := len(kept); i < len(u.live); i++ {
		u.live[i] = nil
	}
	u.live = kept
}

func (u *unionGrammar) CanEnd() bool {
	for _, g := range u.live {
		if g.CanEnd() {
			return true
		}
	}
	return false
}

// InPlainString reports true only when EVERY live branch is inside a plain JSON string: the
// masker's fast path then treats a plain-string-safe token as legal without a walk, which is
// sound for the union only if it is sound for each branch that could still accept it.
func (u *unionGrammar) InPlainString() bool {
	if len(u.live) == 0 {
		return false
	}
	for _, g := range u.live {
		if !inPlainString(g) {
			return false
		}
	}
	return true
}

// Live reports how many branches are still consistent with the committed output — 1 once
// the tool is decided, 0 only if a Commit was forced past every branch (a caller bug: Commit
// requires a prior TryBytes success).
func (u *unionGrammar) Live() int { return len(u.live) }

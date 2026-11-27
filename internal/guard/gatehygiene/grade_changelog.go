// Package gatehygiene implements repository, diff, and baseline-ledger gates below cmd/gate:
// changelog coverage, comments-only diff verification, gate confirmation ledgers, mutation falsifiability,
// and parity-key shadow comparison.
package gatehygiene

import (
	"maps"
	"regexp"
	"slices"
	"strings"
)

// ChangelogFragmentDir holds unreleased changelog fragment markdown files.
const ChangelogFragmentDir = "changelog.d"

const unreleasedPointerPrefix = "Each unreleased change is a file in"

var placeholderRE = regexp.MustCompile(`\{[^}]*\}`)

// NormRoute normalizes `{name}` path parameters in HTTP route patterns to `{}`.
func NormRoute(r string) string { return placeholderRE.ReplaceAllString(r, "{}") }

// ClSection extracts the body of the CHANGELOG section named `Unreleased` or a version tag up to the next `## [`.
func ClSection(doc, name string) (string, bool) {
	head := "## [" + name + "]"
	_, after, ok := strings.Cut(doc, head)
	if !ok {
		return "", false
	}
	after, _, _ = strings.Cut(after, "\n## [")
	return after, true
}

// ClState summarizes user-visible routes, flags, model families, and the aikit pin at a repository revision.
type ClState struct {
	Routes   map[string]bool
	Flags    map[string]bool
	Families map[string][]string
	Pin      string
}

// OwedItem describes one user-visible change omitted from the target CHANGELOG section.
type OwedItem struct{ Kind, Name, Why string }

// ChangelogOwed compares base with head and returns non-waived changes unmentioned in sec.
func ChangelogOwed(base, head *ClState, sec string, waived func(kind, name string) bool) []OwedItem {
	var owed []OwedItem
	add := func(kind, name, why string) {
		if !waived(kind, name) {
			owed = append(owed, OwedItem{kind, name, why})
		}
	}
	nsec := NormRoute(sec)
	var newRoutes []string
	for r := range head.Routes {
		if !base.Routes[r] {
			newRoutes = append(newRoutes, r)
		}
	}
	slices.Sort(newRoutes)
	pathOf := func(r string) string { _, p, _ := strings.Cut(r, " "); return p }
	for _, r := range newRoutes {
		p := pathOf(r)
		if strings.Contains(nsec, p) {
			continue
		}
		covered := false
		for _, o := range newRoutes {
			op := pathOf(o)
			if op != p && strings.HasPrefix(p, strings.TrimSuffix(op, "/")+"/") && strings.Contains(nsec, op) {
				covered = true
				break
			}
		}
		if !covered {
			add("route", r, "a new route no section line names")
		}
	}
	flagRE := func(n string) *regexp.Regexp {
		return regexp.MustCompile(`-{1,2}` + regexp.QuoteMeta(n) + `([^A-Za-z0-9_-]|$)`)
	}
	for _, n := range slices.Sorted(maps.Keys(head.Flags)) {
		if !base.Flags[n] && !flagRE(n).MatchString(sec) {
			add("flag", n, "a new flag")
		}
	}
	for _, n := range slices.Sorted(maps.Keys(base.Flags)) {
		if !head.Flags[n] && !flagRE(n).MatchString(sec) {
			add("flag", n, "a removed flag")
		}
	}
	for _, n := range slices.Sorted(maps.Keys(head.Families)) {
		if _, had := base.Families[n]; had {
			continue
		}
		names := append([]string{n}, head.Families[n]...)
		hit := false
		for _, nm := range names {
			if nm != "" && strings.Contains(strings.ToLower(sec), strings.ToLower(nm)) {
				hit = true
				break
			}
		}
		if !hit {
			add("family", n, "a new model family")
		}
	}
	if head.Pin != "" && head.Pin != base.Pin && !strings.Contains(sec, head.Pin) && !strings.Contains(sec, strings.TrimPrefix(head.Pin, "v")) {
		add("pin", "aikit "+head.Pin, "the aikit pin moved from "+base.Pin)
	}
	return owed
}

func unreleasedStray(body string) []string {
	var out []string
	for ln := range strings.SplitSeq(body, "\n") {
		if strings.TrimSpace(ln) != "" && !strings.HasPrefix(ln, unreleasedPointerPrefix) {
			out = append(out, ln)
		}
	}
	return out
}

package main

import (
	"path/filepath"
	"strings"
)

// A refusal names the rule that matched and what to run instead.
type refusal struct {
	Rule    string
	Message string
}

// Each rule's message names the right command; the rule ids are what the refusal log records and what the monthly rule
// review counts (docs/tasks/task-claude-code-hygiene-2026-10.md, DX4 and DX8).
const (
	msgGA1 = "GA1: never `git add -A`, `git add --all` or `git add .`: it sweeps generated fixture metadata into git, and a " +
		"fixture's skip-guard then reads it as present. Stage explicit paths: `git add <path> <path>…`."
	msgNV1 = "NV1: never skip a git hook (`--no-verify`, `git commit -n`, `-c core.hooksPath=…`). If a hook holds a commit or " +
		"push you believe is wrong, stop and report it to Francis."
	msgSC1 = "SC1: do not run staticcheck with GOOS set by hand: a GOOS=linux `go run` of staticcheck builds a binary this " +
		"machine cannot exec, and the shell still reports exit 0. Run the repo's lint, which sets GOOS for the analysis " +
		"under the pinned binary and proves it analysed: goinfer `go run ./cmd/lint`, tack `go run ./tools/lint`."
	msgTL1 = "TL1: never delete the timing lock (~/.goinfer-timing.lock): its holder keeps its flock on the deleted file, and the " +
		"next run takes a new one beside it, so two timed runs share the box. Run `python3 scripts/timing_lock.py status` " +
		"and wait for the holder."
	msgWT1 = "WT1: do not use the EnterWorktree tool: it writes under .claude/worktrees. Make the worktree with git, outside " +
		"any .claude directory, and work in it by absolute path: `git worktree add <path> -b <branch> origin/main`."
)

// check returns the first refusal for a tool call, or nil to allow it. A command that does not parse is allowed.
func check(tool string, command string) *refusal {
	if tool == "EnterWorktree" {
		return &refusal{"WT1", msgWT1}
	}
	if tool != "Bash" {
		return nil
	}
	segs, ok := commands(command)
	if !ok {
		return nil
	}
	for _, s := range segs {
		for _, r := range []func(segment) *refusal{gitAddAll, noVerify, staticcheckGOOS, timingLockDelete} {
			if f := r(s); f != nil {
				return f
			}
		}
	}
	return nil
}

// gitValueOpts are git's global options that take the next word as their value when not written --opt=value.
var gitValueOpts = map[string]bool{"-C": true, "--git-dir": true, "--work-tree": true, "--namespace": true, "--super-prefix": true}

// gitArgs returns a git invocation's global config values (-c, --config-env) and its subcommand with that subcommand's
// arguments.
func gitArgs(words []string) (configs []string, sub string, args []string, ok bool) {
	if len(words) == 0 || filepath.Base(words[0]) != "git" {
		return nil, "", nil, false
	}
	for i := 1; i < len(words); i++ {
		w := words[i]
		switch {
		case (w == "-c" || w == "--config-env") && i+1 < len(words):
			configs = append(configs, words[i+1])
			i++
		case strings.HasPrefix(w, "-c") && !strings.HasPrefix(w, "--"):
			configs = append(configs, w[2:])
		case strings.HasPrefix(w, "--config-env="):
			configs = append(configs, strings.TrimPrefix(w, "--config-env="))
		case gitValueOpts[w]:
			i++
		case strings.HasPrefix(w, "-"):
		default:
			return configs, w, words[i+1:], true
		}
	}
	return configs, "", nil, true
}

func gitAddAll(s segment) *refusal {
	_, sub, args, ok := gitArgs(s.words)
	if !ok || sub != "add" {
		return nil
	}
	paths := false
	for _, a := range args {
		switch {
		case paths || !strings.HasPrefix(a, "-"):
			if a == "." || a == "./" || a == ":/" || a == ":." {
				return &refusal{"GA1", msgGA1}
			}
		case a == "--":
			paths = true
		case a == "-A" || a == "--all" || a == "--no-ignore-removal":
			return &refusal{"GA1", msgGA1}
		case !strings.HasPrefix(a, "--") && strings.Contains(a[1:], "A"):
			return &refusal{"GA1", msgGA1}
		}
	}
	return nil
}

// gitValues lists, per subcommand NV1 reads, the short options (by letter) and long options that take a value, so a value
// that looks like a flag (`git commit -m --no-verify`) is read as the value it is.
var gitValues = map[string]struct {
	short string
	long  map[string]bool
}{
	"commit": {"mFCctS", map[string]bool{"--message": true, "--file": true, "--author": true, "--date": true, "--template": true,
		"--reuse-message": true, "--reedit-message": true, "--fixup": true, "--squash": true, "--cleanup": true, "--trailer": true,
		"--pathspec-from-file": true}},
	"merge":  {"mFsX", map[string]bool{"--message": true, "--file": true, "--strategy": true, "--strategy-option": true, "--cleanup": true}},
	"pull":   {"sX", map[string]bool{"--strategy": true, "--strategy-option": true, "--depth": true}},
	"push":   {"o", map[string]bool{"--push-option": true, "--repo": true, "--receive-pack": true, "--exec": true}},
	"rebase": {"sXx", map[string]bool{"--strategy": true, "--strategy-option": true, "--exec": true, "--onto": true}},
	"am":     {"pC", map[string]bool{"--directory": true, "--exclude": true, "--include": true, "--patch-format": true}},
}

// hookSubcommands are the git subcommands whose --no-verify skips a hook. Elsewhere (git grep, git log -S) the string is
// a search term.
var hookSubcommands = map[string]bool{"commit": true, "push": true, "merge": true, "rebase": true, "am": true, "pull": true,
	"cherry-pick": true, "revert": true}

func noVerify(s segment) *refusal {
	configs, sub, args, ok := gitArgs(s.words)
	if !ok {
		return nil
	}
	for _, c := range configs {
		if strings.HasPrefix(strings.ToLower(c), "core.hookspath") {
			return &refusal{"NV1", msgNV1}
		}
	}
	if !hookSubcommands[sub] {
		return nil
	}
	vals := gitValues[sub]
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--":
			return nil
		case a == "--no-verify":
			return &refusal{"NV1", msgNV1}
		case strings.HasPrefix(a, "--"):
			if !strings.Contains(a, "=") && vals.long[a] {
				i++
			}
		case len(a) > 1 && a[0] == '-':
			for k := 1; k < len(a); k++ {
				if sub == "commit" && a[k] == 'n' {
					return &refusal{"NV1", msgNV1}
				}
				if strings.IndexByte(vals.short, a[k]) >= 0 {
					if k == len(a)-1 {
						i++
					}
					break
				}
			}
		}
	}
	return nil
}

func staticcheckGOOS(s segment) *refusal {
	if _, set := s.env["GOOS"]; !set || len(s.words) == 0 {
		return nil
	}
	if filepath.Base(s.words[0]) == "staticcheck" {
		return &refusal{"SC1", msgSC1}
	}
	if filepath.Base(s.words[0]) == "go" && len(s.words) > 2 && s.words[1] == "run" {
		for _, w := range s.words[2:] {
			if strings.Contains(w, "staticcheck") {
				return &refusal{"SC1", msgSC1}
			}
		}
	}
	return nil
}

var deleters = map[string]bool{"rm": true, "unlink": true, "mv": true, "trash": true, "shred": true, "srm": true}

func timingLockDelete(s segment) *refusal {
	if len(s.words) == 0 || !deleters[filepath.Base(s.words[0])] {
		return nil
	}
	for _, w := range s.words[1:] {
		if strings.Contains(w, ".goinfer-timing.lock") {
			return &refusal{"TL1", msgTL1}
		}
	}
	return nil
}

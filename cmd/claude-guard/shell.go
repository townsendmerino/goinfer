package main

import (
	"path/filepath"
	"regexp"
	"strings"
)

// A segment is one simple command: its leading NAME=value assignments and its words, with quotes removed. A quoted
// argument is one word, so a commit message that quotes a refused command is an argument, never a command.
type segment struct {
	env   map[string]string
	words []string
}

var heredocRe = regexp.MustCompile(`(?:^|[^<])<<-?\s*(['"]?)([A-Za-z_][A-Za-z0-9_]*)['"]?`)

// stripHeredocs drops heredoc bodies: they are stdin data, not commands the shell runs.
func stripHeredocs(s string) string {
	lines := strings.Split(s, "\n")
	var out []string
	for i := 0; i < len(lines); i++ {
		out = append(out, lines[i])
		for _, m := range heredocRe.FindAllStringSubmatch(lines[i], -1) {
			for i++; i < len(lines) && strings.TrimSpace(lines[i]) != m[2]; i++ {
			}
		}
	}
	return strings.Join(out, "\n")
}

// split lexes a command line into segments at ; & | && || newlines and parentheses. ok is false on an unterminated
// quote, and the caller then allows the command (the hook fails open).
func split(s string) (segs [][]string, ok bool) {
	s = stripHeredocs(s)
	var cur []string
	var word strings.Builder
	inWord := false
	flushWord := func() {
		if inWord {
			cur = append(cur, word.String())
			word.Reset()
			inWord = false
		}
	}
	flushSeg := func() {
		flushWord()
		if len(cur) > 0 {
			segs = append(segs, cur)
			cur = nil
		}
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '\'':
			j := strings.IndexByte(s[i+1:], '\'')
			if j < 0 {
				return nil, false
			}
			word.WriteString(s[i+1 : i+1+j])
			inWord = true
			i += j + 1
		case c == '"':
			i++
			for ; i < len(s) && s[i] != '"'; i++ {
				if s[i] == '\\' && i+1 < len(s) && strings.IndexByte("\"\\$`\n", s[i+1]) >= 0 {
					i++
				}
				word.WriteByte(s[i])
			}
			if i >= len(s) {
				return nil, false
			}
			inWord = true
		case c == '\\' && i+1 < len(s):
			i++
			if s[i] != '\n' {
				word.WriteByte(s[i])
				inWord = true
			}
		case c == '#' && !inWord:
			for i < len(s) && s[i] != '\n' {
				i++
			}
			flushSeg()
		case strings.IndexByte(";&|()\n", c) >= 0:
			flushSeg()
		case c == ' ' || c == '\t':
			flushWord()
		default:
			word.WriteByte(c)
			inWord = true
		}
	}
	flushSeg()
	return segs, true
}

var assignRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=`)

// wrappers run the rest of their words as a command; the value is how many words after the wrapper's own name are its
// arguments before the command starts (flags are skipped separately).
var wrappers = map[string]int{"sudo": 0, "time": 0, "nohup": 0, "command": 0, "exec": 0, "nice": 0, "caffeinate": 0, "timeout": 1}

// commands expands a command line into the segments it runs: assignments are lifted off each segment, wrappers (env,
// sudo, timeout, `timing_lock.py run … --`) are unwrapped, and `bash -c '…'` and `ssh host '…'` are parsed inside, to
// depth 4. An `export NAME=value` applies to the segments after it.
func commands(s string) ([]segment, bool) {
	return commandsDepth(s, 0, map[string]string{})
}

func commandsDepth(s string, depth int, exported map[string]string) ([]segment, bool) {
	segs, ok := split(s)
	if !ok || depth > 4 {
		return nil, ok
	}
	var out []segment
	for _, words := range segs {
		seg := segment{env: map[string]string{}}
		for k, v := range exported {
			seg.env[k] = v
		}
		words = liftEnv(words, seg.env)
		for {
			if len(words) == 0 {
				break
			}
			base := filepath.Base(words[0])
			if base == "env" {
				words = words[1:]
				for len(words) > 0 && strings.HasPrefix(words[0], "-") {
					if words[0] == "-u" && len(words) > 1 {
						words = words[1:]
					}
					words = words[1:]
				}
				words = liftEnv(words, seg.env)
				continue
			}
			if n, ok := wrappers[base]; ok {
				words = words[1:]
				for len(words) > 0 && strings.HasPrefix(words[0], "-") {
					words = words[1:]
				}
				if n <= len(words) {
					words = words[n:]
				}
				continue
			}
			if strings.HasSuffix(base, "timing_lock.py") || (strings.HasPrefix(base, "python") && len(words) > 1 && strings.HasSuffix(words[1], "timing_lock.py")) {
				if i := indexOf(words, "--"); i >= 0 {
					words = words[i+1:]
					continue
				}
			}
			break
		}
		if len(words) == 0 {
			continue
		}
		base := filepath.Base(words[0])
		if base == "export" {
			for _, w := range words[1:] {
				if assignRe.MatchString(w) {
					k, v, _ := strings.Cut(w, "=")
					exported[k] = v
				}
			}
			continue
		}
		if inner, ok := shellScript(base, words); ok {
			sub, _ := commandsDepth(inner, depth+1, copyEnv(seg.env))
			out = append(out, sub...)
			continue
		}
		if base == "ssh" {
			if remote := sshRemote(words); remote != "" {
				sub, _ := commandsDepth(remote, depth+1, map[string]string{})
				out = append(out, sub...)
			}
			continue
		}
		seg.words = words
		out = append(out, seg)
	}
	return out, true
}

func liftEnv(words []string, env map[string]string) []string {
	for len(words) > 0 && assignRe.MatchString(words[0]) {
		k, v, _ := strings.Cut(words[0], "=")
		env[k] = v
		words = words[1:]
	}
	return words
}

func copyEnv(m map[string]string) map[string]string {
	c := make(map[string]string, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}

func indexOf(words []string, w string) int {
	for i, x := range words {
		if x == w {
			return i
		}
	}
	return -1
}

// shellScript returns the script of `bash|sh|zsh|dash|ksh [flags] -c SCRIPT` (the c may share a flag cluster: -lc, -ec).
func shellScript(base string, words []string) (string, bool) {
	switch base {
	case "bash", "sh", "zsh", "dash", "ksh":
	default:
		return "", false
	}
	for i := 1; i < len(words); i++ {
		w := words[i]
		if !strings.HasPrefix(w, "-") || strings.HasPrefix(w, "--") {
			return "", false
		}
		if strings.Contains(w, "c") && i+1 < len(words) {
			return words[i+1], true
		}
	}
	return "", false
}

// sshArgOpts are ssh's options that take a value.
const sshArgOpts = "BbcDEeFIiJLlmOopQRSWw"

// sshRemote returns the command an ssh invocation runs on the remote host, joined as the remote shell will see it.
func sshRemote(words []string) string {
	i := 1
	for i < len(words) && strings.HasPrefix(words[i], "-") {
		w := words[i]
		i++
		if len(w) == 2 && strings.IndexByte(sshArgOpts, w[1]) >= 0 {
			i++
		}
	}
	if i+1 >= len(words) {
		return ""
	}
	return strings.Join(words[i+1:], " ")
}

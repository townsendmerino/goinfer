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

// lexer splits a command line into simple commands the way a shell does, closely enough to find the commands it runs:
// quotes, backslashes and comments; ; & | && || newlines and parentheses as separators (but not the & of a 2>&1 or &>
// redirection); heredoc bodies, which are data, skipped from the line after their unquoted << operator; and $( ) and
// backtick substitutions, whose commands run too and are lexed as commands of their own.
type lexer struct {
	s       string
	i       int
	segs    [][]string
	pending []heredoc
	ok      bool
}

type heredoc struct {
	delim string
	tabs  bool // <<-: leading tabs are stripped from body lines
}

// split lexes s into its simple commands. ok is false on an unterminated quote or substitution, and the caller then
// allows the command (the hook fails open).
func split(s string) ([][]string, bool) {
	lx := &lexer{s: s, ok: true}
	lx.run(false)
	return lx.segs, lx.ok
}

// run lexes until the end of s, or, when inSub, until the ) that closes a $( substitution (consumed).
func (lx *lexer) run(inSub bool) {
	var cur []string
	var word strings.Builder
	inWord, depth := false, 0
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
			lx.segs = append(lx.segs, cur)
			cur = nil
		}
	}
	defer flushSeg()
	s := lx.s
	for lx.i < len(s) {
		c := s[lx.i]
		switch {
		case c == '\'':
			j := strings.IndexByte(s[lx.i+1:], '\'')
			if j < 0 {
				lx.ok = false
				lx.i = len(s)
				return
			}
			word.WriteString(s[lx.i+1 : lx.i+1+j])
			inWord = true
			lx.i += j + 2
		case c == '"':
			lx.i++
			lx.doubleQuoted(&word)
			inWord = true
		case c == '\\' && lx.i+1 < len(s):
			if s[lx.i+1] != '\n' {
				word.WriteByte(s[lx.i+1])
				inWord = true
			}
			lx.i += 2
		case c == '$' && strings.HasPrefix(s[lx.i:], "$(("):
			lx.arithmetic(&word)
			inWord = true
		case c == '$' && strings.HasPrefix(s[lx.i:], "$("):
			lx.i += 2
			lx.substitution()
			word.WriteString("$(…)")
			inWord = true
		case c == '`':
			lx.backtick()
			word.WriteString("`…`")
			inWord = true
		case c == '#' && !inWord:
			for lx.i < len(s) && s[lx.i] != '\n' {
				lx.i++
			}
		case c == '<' && strings.HasPrefix(s[lx.i:], "<<<"):
			flushWord()
			lx.i += 3
		case c == '<' && strings.HasPrefix(s[lx.i:], "<<"):
			flushWord()
			lx.i += 2
			lx.heredocOperator()
		case c == '\n':
			flushSeg()
			lx.i++
			lx.skipHeredocBodies()
		case c == '&' && (lx.i > 0 && (s[lx.i-1] == '>' || s[lx.i-1] == '<') || strings.HasPrefix(s[lx.i:], "&>")):
			word.WriteByte(c) // part of a redirection: 2>&1, <&3, &>file
			inWord = true
			lx.i++
		case c == '(':
			flushSeg()
			depth++
			lx.i++
		case c == ')':
			flushSeg()
			lx.i++
			if depth == 0 && inSub {
				return
			}
			if depth > 0 {
				depth--
			}
		case c == ';' || c == '&' || c == '|':
			flushSeg()
			lx.i++
		case c == ' ' || c == '\t':
			flushWord()
			lx.i++
		default:
			word.WriteByte(c)
			inWord = true
			lx.i++
		}
	}
	if inSub {
		lx.ok = false
	}
}

// doubleQuoted reads a double-quoted string's body after its opening quote, into word; substitutions inside it are
// lexed as commands.
func (lx *lexer) doubleQuoted(word *strings.Builder) {
	s := lx.s
	for lx.i < len(s) {
		c := s[lx.i]
		switch {
		case c == '"':
			lx.i++
			return
		case c == '\\' && lx.i+1 < len(s) && strings.IndexByte("\"\\$`\n", s[lx.i+1]) >= 0:
			word.WriteByte(s[lx.i+1])
			lx.i += 2
		case c == '$' && strings.HasPrefix(s[lx.i:], "$(("):
			lx.arithmetic(word)
		case c == '$' && strings.HasPrefix(s[lx.i:], "$("):
			lx.i += 2
			lx.substitution()
			word.WriteString("$(…)")
		case c == '`':
			lx.backtick()
			word.WriteString("`…`")
		default:
			word.WriteByte(c)
			lx.i++
		}
	}
	lx.ok = false
}

// substitution lexes the commands of a $( ) whose $( has been consumed, through its closing ).
func (lx *lexer) substitution() {
	sub := &lexer{s: lx.s, i: lx.i, ok: true}
	sub.run(true)
	lx.segs = append(lx.segs, sub.segs...)
	lx.i = sub.i
	lx.ok = lx.ok && sub.ok
}

// backtick lexes the commands of a `…` substitution, through its closing backtick.
func (lx *lexer) backtick() {
	s := lx.s
	j := lx.i + 1
	var body strings.Builder
	for ; j < len(s) && s[j] != '`'; j++ {
		if s[j] == '\\' && j+1 < len(s) {
			j++
		}
		body.WriteByte(s[j])
	}
	if j >= len(s) {
		lx.ok = false
		lx.i = len(s)
		return
	}
	segs, ok := split(body.String())
	lx.segs = append(lx.segs, segs...)
	lx.ok = lx.ok && ok
	lx.i = j + 1
}

// arithmetic copies a $(( )) expression into word: its << is a shift, not a heredoc.
func (lx *lexer) arithmetic(word *strings.Builder) {
	s := lx.s
	depth := 0
	for lx.i < len(s) {
		c := s[lx.i]
		word.WriteByte(c)
		lx.i++
		if c == '(' {
			depth++
		} else if c == ')' {
			depth--
			if depth == 0 {
				return
			}
		}
	}
	lx.ok = false
}

// heredocOperator reads the delimiter after an unquoted << (consumed): <<EOF, <<-EOF, <<'EOF', <<"EOF", <<\EOF.
func (lx *lexer) heredocOperator() {
	s := lx.s
	h := heredoc{}
	if lx.i < len(s) && s[lx.i] == '-' {
		h.tabs = true
		lx.i++
	}
	for lx.i < len(s) && (s[lx.i] == ' ' || s[lx.i] == '\t') {
		lx.i++
	}
	var d strings.Builder
	for lx.i < len(s) && strings.IndexByte(" \t\n;&|()<>", s[lx.i]) < 0 {
		switch c := s[lx.i]; c {
		case '\'', '"':
			j := strings.IndexByte(s[lx.i+1:], c)
			if j < 0 {
				lx.ok = false
				lx.i = len(s)
				return
			}
			d.WriteString(s[lx.i+1 : lx.i+1+j])
			lx.i += j + 2
		case '\\':
			if lx.i+1 < len(s) {
				d.WriteByte(s[lx.i+1])
			}
			lx.i += 2
		default:
			d.WriteByte(c)
			lx.i++
		}
	}
	if d.Len() > 0 {
		lx.pending = append(lx.pending, heredoc{delim: d.String(), tabs: h.tabs})
	}
}

// skipHeredocBodies skips, after a newline, the body of each heredoc opened on the line just ended. A body ends at its
// delimiter alone on a line, or at the delimiter followed by the ) that closes the substitution it sits in.
func (lx *lexer) skipHeredocBodies() {
	s := lx.s
	for _, h := range lx.pending {
		for lx.i < len(s) {
			end := strings.IndexByte(s[lx.i:], '\n')
			line := s[lx.i:]
			if end >= 0 {
				line = s[lx.i : lx.i+end]
			}
			t := line
			if h.tabs {
				t = strings.TrimLeft(t, "\t")
			}
			if t == h.delim {
				lx.i += len(line)
				if end >= 0 {
					lx.i++
				}
				break
			}
			if strings.HasPrefix(t, h.delim+")") {
				lx.i += len(line) - len(t) + len(h.delim)
				break
			}
			if end < 0 {
				lx.i = len(s)
				break
			}
			lx.i += end + 1
		}
	}
	lx.pending = nil
}

var assignRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=`)

// reserved are shell words that can precede a command in the same simple-command position.
var reserved = map[string]bool{"{": true, "}": true, "!": true, "if": true, "then": true, "else": true, "elif": true,
	"do": true, "while": true, "until": true, "fi": true, "done": true}

// wrappers run the rest of their words as a command. The value lists the short options that take a value, and positional
// is how many words after the options are the wrapper's own (timeout's duration) before the command starts.
var wrappers = map[string]struct {
	argOpts    string
	positional int
}{
	"sudo": {"ugCDhpUrtT", 0}, "env": {"uCS", 0}, "nice": {"n", 0}, "timeout": {"sk", 1}, "nohup": {"", 0},
	"time": {"fo", 0}, "command": {"", 0}, "exec": {"a", 0}, "caffeinate": {"tw", 0},
}

// skipOptions drops the leading options of an argument list, consuming the value of an option in argOpts (as the next
// word, or the rest of its cluster). Long options are taken as --name or --name=value. It stops at -- (consumed).
func skipOptions(words []string, argOpts string) []string {
	for len(words) > 0 {
		w := words[0]
		if w == "--" {
			return words[1:]
		}
		if !strings.HasPrefix(w, "-") || w == "-" {
			return words
		}
		words = words[1:]
		if strings.HasPrefix(w, "--") {
			continue
		}
		for k := 1; k < len(w); k++ {
			if strings.IndexByte(argOpts, w[k]) >= 0 {
				if k == len(w)-1 && len(words) > 0 {
					words = words[1:]
				}
				break
			}
		}
	}
	return words
}

// commands expands a command line into the segments it runs: assignments are lifted off each segment, reserved words and
// wrappers (env, sudo, timeout, `timing_lock.py run … --`) are unwrapped, and `bash -c '…'` and `ssh host …` are parsed
// inside, to depth 4. An `export NAME=value` applies to the segments after it.
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
		seg := segment{env: copyEnv(exported)}
		for len(words) > 0 {
			words = liftEnv(words, seg.env)
			if len(words) == 0 {
				break
			}
			base := filepath.Base(words[0])
			if reserved[words[0]] {
				words = words[1:]
				continue
			}
			if w, ok := wrappers[base]; ok {
				words = skipOptions(words[1:], w.argOpts)
				if w.positional <= len(words) {
					words = words[w.positional:]
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

// shellScript returns the script of `bash|sh|zsh|dash|ksh [options] -c SCRIPT`: the c may share a cluster (-lc, -ec),
// long options (--norc, --login) and -o/+o pairs may come first.
func shellScript(base string, words []string) (string, bool) {
	switch base {
	case "bash", "sh", "zsh", "dash", "ksh":
	default:
		return "", false
	}
	for i := 1; i < len(words); i++ {
		w := words[i]
		switch {
		case w == "--":
			return "", false
		case w == "--rcfile" || w == "--init-file":
			i++
		case strings.HasPrefix(w, "--"):
		case w == "-o" || w == "+o" || w == "-O" || w == "+O":
			i++
		case strings.HasPrefix(w, "-") || strings.HasPrefix(w, "+"):
			if strings.Contains(w[1:], "c") && i+1 < len(words) {
				return words[i+1], true
			}
		default:
			return "", false // a script file, not -c
		}
	}
	return "", false
}

// sshArgOpts are ssh's options that take a value.
const sshArgOpts = "BbcDEeFIiJLlmOopQRSWw"

// sshRemote returns the command an ssh invocation runs on the remote host. ssh joins its remaining words with spaces and
// the remote shell splits that string again, so the join is what runs there: `ssh host git commit -m "a b"` sends
// `git commit -m a b`.
func sshRemote(words []string) string {
	rest := skipOptions(words[1:], sshArgOpts)
	if len(rest) < 2 {
		return ""
	}
	return strings.Join(rest[1:], " ")
}

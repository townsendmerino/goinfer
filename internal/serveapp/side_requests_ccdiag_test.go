package serveapp

import (
	"compress/gzip"
	"fmt"
	"flag"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestSideRequests_ccDiag renders Claude Code requests rebuilt from an agent's own transcripts (cc_rebuild.py beside the e2e record) the way serve does,
// and reports where they diverge and what sits there. A diagnostic: it asserts nothing and skips without CCDIAG (the directory of rebuilt bodies; CCEXTRA
// names more bodies in it, compared with the edit session's start).
func TestSideRequests_ccDiag(t *testing.T) {
	// The log lines below quote decoded prompt text around each divergence. A request body of a real user's session holds that user's files (their
	// CLAUDE.md, their memory index), and this repository is public: the text is shown only with CCDIAG_TEXT=1, and output made that way is not for a commit.
	show := func(lm *loadedModel, ids []int) string {
		if os.Getenv("CCDIAG_TEXT") != "1" {
			return fmt.Sprintf("<%d ids; set CCDIAG_TEXT=1 to decode>", len(ids))
		}
		return renderText(lm, ids)
	}
	dir := os.Getenv("CCDIAG")
	if dir == "" {
		t.Skip("CCDIAG")
	}
	tk, err := modelload.Tokenizer(filepath.Join("..", "..", "testdata", "mellum2-tokenizer"))
	if err != nil {
		t.Fatal(err)
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	sf := registerFlags(fs)
	_ = fs.Parse(nil)
	s := &server{cfg: sf.cfg}
	lm := &loadedModel{tk: tk, name: "mellum2.1-thinking"}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	ids := map[string][]int{}
	for _, n := range []string{"read", "search", "edit"} {
		f, _ := os.Open(filepath.Join(dir, n+".req.json.gz"))
		zr, _ := gzip.NewReader(f)
		raw, _ := io.ReadAll(zr)
		f.Close()
		x, _, err := kv0RenderIDs(s, lm, "/v1/messages", raw, false)
		if err != nil {
			t.Fatal(err)
		}
		ids[n] = x
		t.Logf("%s: %d ids", n, len(x))
	}
	// first index at which the decoded prefix contains marker
	at := func(x []int, marker string) int {
		lo, hi := 0, len(x)
		for lo < hi {
			m := (lo + hi) / 2
			if strings.Contains(renderText(lm, x[:m]), marker) {
				hi = m
			} else {
				lo = m + 1
			}
		}
		return lo
	}
	pairs := [][2]string{{"read", "search"}, {"search", "edit"}, {"read", "edit"}}
	for _, p := range pairs {
		a, b := ids[p[0]], ids[p[1]]
		l := idsLCP(a, b)
		t.Logf("%s vs %s: LCP %d; %s | ... shared ends: %q  / next in %s: %q  / next in %s: %q; ids after the divergence: %d and %d",
			p[0], p[1], l, "", show(lm, a[max(0, l-12):l]), p[0], show(lm, a[l:min(len(a), l+12)]), p[1], show(lm, b[l:min(len(b), l+12)]), len(a)-l, len(b)-l)
	}
	ids2 := map[string][]int{}
	for _, n := range strings.Split(os.Getenv("CCEXTRA"), ",") {
		if n == "" {
			continue
		}
		f, _ := os.Open(filepath.Join(dir, n+".req.json.gz"))
		zr, _ := gzip.NewReader(f)
		raw, _ := io.ReadAll(zr)
		f.Close()
		x, _, err := kv0RenderIDs(s, lm, "/v1/messages", raw, false)
		if err != nil {
			t.Fatal(err)
		}
		ids2[n] = x
		l := idsLCP(ids["edit"], x)
		t.Logf("%s: %d ids; LCP with the edit start %d; shared ends %q / next in edit %q / next in %s %q", n, len(x), l,
			show(lm, x[max(0, l-8):l]), show(lm, ids["edit"][l:l+14]), n, show(lm, x[l:min(len(x), l+14)]))
		for _, m := range []string{"This session is being continued", "Today's date", "Available agent types", "# Environment", "You are powered by", "README.md"} {
			t.Logf("  %s marker %q first appears by id %d", n, m, at(x, m))
		}
		if n == "post2B" {
			l1 := idsLCP(ids2["post1B"], x)
			t.Logf("  post2B vs post1B: LCP %d (logged div for the request after the second compaction, 16,420, plus the rebuild's constant offset: 49 with the compact serialization)", l1)
		}

	}
	e := ids["edit"]
	for _, m := range []string{"Recent commits:", "# Environment", "As you answer the user", "Primary working directory", "Available agent types for the Agent tool", "Rename `sum`", "Attribution for git commits", "Available agent types", "skill", "Today's date is", "Today's date"} {
		t.Logf("marker %q first appears by id %d of %d", m, at(e, m), len(e))
	}
}

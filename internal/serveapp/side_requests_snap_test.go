package serveapp

import (
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestSideRequests_snapshotPrefixes checks a run's restored prefixes against the ids the run itself wrote (KV7 step 2, the night rerun with bodies kept,
// docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md §8). SIDE_OUT is a matrix.py output directory, SIDE_CONFIG the configuration (its serve log is
// logs/serve-<name>.log and its request bodies bodies-<name>/NNNNN.req.json.gz, in the order of the log's request lines). For every request whose line
// carries snap=<rows>, the first <rows> ids of its rendered prompt must be the first <rows> ids of an EARLIER request's rendered prompt: a restore puts
// back rows a prefill wrote for those ids, and nothing else is a legitimate source. It also reports, per request, the rows reused, the time to first token
// and whether it restored. A diagnostic that asserts that one thing; it skips without SIDE_OUT.
func TestSideRequests_snapshotPrefixes(t *testing.T) {
	out, name := os.Getenv("SIDE_OUT"), os.Getenv("SIDE_CONFIG")
	if out == "" {
		t.Skip("SIDE_OUT names a matrix.py output directory to read")
	}
	if name == "" {
		t.Fatal("SIDE_CONFIG names the configuration")
	}
	tk, err := modelload.Tokenizer(filepath.Join("..", "..", "testdata", "mellum2-tokenizer"))
	if err != nil {
		t.Skipf("needs the Mellum2 tokenizer: %v", err)
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	sf := registerFlags(fs)
	_ = fs.Parse(nil)
	s := &server{cfg: sf.cfg}
	lm := &loadedModel{tk: tk, name: "mellum2.1-thinking"}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	re := regexp.MustCompile(`^request: POST (\S+) .*prompt_tokens=(\d+) .*ttft=(\d+)ms .*reused=(\S+) .*?(?: snap=(\d+))?$`)
	logRaw, err := os.ReadFile(filepath.Join(out, "logs", "serve-"+name+".log"))
	if err != nil {
		t.Fatal(err)
	}
	var prompts [][]int
	var bad int
	i := 0
	for _, l := range strings.Split(string(logRaw), "\n") {
		m := re.FindStringSubmatch(l)
		if m == nil {
			continue
		}
		i++
		f, err := os.Open(filepath.Join(out, "bodies-"+name, fmt.Sprintf("%05d.req.json.gz", i)))
		if err != nil {
			t.Fatal(err)
		}
		zr, _ := gzip.NewReader(f)
		raw, _ := io.ReadAll(zr)
		f.Close()
		route := strings.SplitN(m[1], "?", 2)[0]
		ids, _, err := kv0RenderIDs(s, lm, route, raw, false)
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		ttft, _ := strconv.Atoi(m[3])
		snap, _ := strconv.Atoi(m[5])
		best := 0
		for _, p := range prompts {
			best = max(best, idsLCP(ids, p))
		}
		verdict := "cold or continued"
		if snap > 0 {
			verdict = "restored"
			if best < snap {
				verdict = "RESTORED ROWS NO EARLIER PROMPT WROTE"
				bad++
			}
		}
		t.Logf("#%d %s %d ids, %.1f s to first token, reused %s, snap %d, longest shared prefix with an earlier prompt %d: %s", i, route, len(ids), float64(ttft)/1000, m[4], snap, best, verdict)
		prompts = append(prompts, ids)
	}
	if i == 0 {
		t.Fatal("no request lines")
	}
	if bad > 0 {
		t.Errorf("%d restores put back rows that no earlier prompt wrote", bad)
	}
}

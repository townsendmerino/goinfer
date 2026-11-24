package serveapp

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/app/loadflags"
)

// A load that panics from the web UI ends in an error event on that load's stream, releases the single-flight, and leaves the server
// serving: the next load runs.
func TestPanicGuard_webLoadPanicIsAnErrorEventAndTheServerKeepsServing(t *testing.T) {
	_, root := fakeCache(t)
	p := pulledModel(t, root, "boom.gguf")
	other := pulledModel(t, root, "fine.gguf")
	s, err := newServer(config{web: true, load: loadflags.Flags{Backend: "cpu"}})
	if err != nil {
		t.Fatal(err)
	}
	orig := webLoadDecoder
	t.Cleanup(func() { webLoadDecoder = orig })
	webLoadDecoder = func(_ context.Context, spec modelSpec, _ config) (*loadedModel, error) {
		if spec.name == "boom" {
			var counts []int
			counts[len(spec.name)]++ // a bug in the loader: index out of range
		}
		return &loadedModel{name: spec.name}, nil
	}

	evs := readLoadEvents(t, postLoad(s, context.Background(), p).Body)
	if len(evs) == 0 || evs[len(evs)-1].name != "error" {
		t.Fatalf("events after a panicking load = %+v, want the stream to end in an error event", evs)
	}
	if msg, _ := evs[len(evs)-1].data["message"].(string); !strings.Contains(msg, "internal error") {
		t.Errorf("error message = %q, want the generic internal error", msg)
	}
	evs = readLoadEvents(t, postLoad(s, context.Background(), other).Body)
	if len(evs) == 0 || evs[len(evs)-1].name != "done" {
		t.Errorf("load after the panic = %+v, want it to run to done (single-flight released)", evs)
	}
}

// An SSE heartbeat whose write panics ends without taking the process down, and its stop still returns.
func TestPanicGuard_sseHeartbeatPanicEndsTheHeartbeatOnly(t *testing.T) {
	old := sseHeartbeatInterval
	sseHeartbeatInterval = time.Millisecond
	t.Cleanup(func() { sseHeartbeatInterval = old })
	stop := sseHeartbeat(&sseWriter{}) // no ResponseWriter: the first ping dereferences nil
	time.Sleep(20 * time.Millisecond)
	stop()
}

// A drain whose checkpoint or free panics closes its done channel and leaves the model marked draining.
func TestPanicGuard_startDrainPanicKeepsTheModelMarkedDraining(t *testing.T) {
	s := &server{cfg: config{sessionDir: t.TempDir(), kvSessions: 1}}
	lm := &loadedModel{name: "m"} // no session LRU: the checkpoint dereferences nil
	select {
	case <-s.startDrain(lm, nil, true):
	case <-time.After(10 * time.Second):
		t.Fatal("the drain's done channel never closed after a panic")
	}
	s.regMu.RLock()
	_, still := s.draining["m"]
	s.regMu.RUnlock()
	if !still {
		t.Error("a drain that panicked cleared its draining mark: /health would report the memory freed")
	}
}

// goroutineGuardExempt names the functions in this package whose `go func` literals take no panic guard. None runs request work.
var goroutineGuardExempt = map[string]string{
	"startAdminSocket":    "serves the admin socket's own http.Server, whose handlers net/http recovers",
	"startHaltSignalLoop": "the SIGUSR1/SIGUSR2 loop: process control, not request work",
	"Main":                "the shutdown signal handlers: they end the process on purpose",
}

// Every `go func` literal in the package's non-test files defers a recover as one of its top-level statements: a deferred func
// literal that calls recover(), or a call to a recover* function (panicguard.go). A handler-spawned goroutine without one ends the
// process on a panic.
func TestPanicGuard_everyGoFuncLiteralDefersARecover(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var missing []string
	exemptSeen := map[string]bool{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		file, err := parser.ParseFile(fset, f, src, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range file.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				g, ok := n.(*ast.GoStmt)
				if !ok {
					return true
				}
				lit, ok := g.Call.Fun.(*ast.FuncLit)
				if !ok || defersRecover(lit.Body) {
					return true
				}
				if _, ok := goroutineGuardExempt[fd.Name.Name]; ok {
					exemptSeen[fd.Name.Name] = true
					return true
				}
				missing = append(missing, fset.Position(g.Pos()).Filename+": in "+fd.Name.Name)
				return true
			})
		}
	}
	slices.Sort(missing)
	for _, m := range missing {
		t.Errorf("%s: a `go func` with no deferred recover; defer one of panicguard.go's recover* functions, or a func literal that recovers", m)
	}
	for name := range goroutineGuardExempt {
		if !exemptSeen[name] {
			t.Errorf("goroutineGuardExempt names %s, which no longer starts an unguarded goroutine: remove the entry", name)
		}
	}
}

func defersRecover(body *ast.BlockStmt) bool {
	for _, st := range body.List {
		d, ok := st.(*ast.DeferStmt)
		if !ok {
			continue
		}
		switch fn := d.Call.Fun.(type) {
		case *ast.Ident:
			if strings.HasPrefix(fn.Name, "recover") {
				return true
			}
		case *ast.SelectorExpr:
			if strings.HasPrefix(fn.Sel.Name, "recover") {
				return true
			}
		case *ast.FuncLit:
			found := false
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if c, ok := n.(*ast.CallExpr); ok {
					if id, ok := c.Fun.(*ast.Ident); ok && id.Name == "recover" {
						found = true
					}
				}
				return !found
			})
			if found {
				return true
			}
		}
	}
	return false
}

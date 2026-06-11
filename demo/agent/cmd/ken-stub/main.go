// cmd/ken-stub is a throwaway MCP `search` server that returns canned go-stdlib
// snippets, so you can play the agent's DECIDE → search → ANSWER loop with just a
// model — no ken index build, no ken repo. It is NOT retrieval: it returns the
// same few file:line chunks for every query, enough to exercise the wiring (the
// constrained tool-call, the MCP round-trip, the grounded/cited answer phase).
//
// Build & use:
//
//	GOWORK=off go build -o /tmp/ken-stub ./cmd/ken-stub
//	GOWORK=off go run ./cmd/agent-web --model <model.gguf> --ken /tmp/ken-stub
package main

import (
	"context"
	"log"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// searchArgs matches the agent's tools/call arguments ({query, top_k}).
type searchArgs struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

// cannedChunks are a few plausible go-stdlib snippets (file:line cited, the
// markdown shape the agent splices into the answer prompt). They cover the
// README's opening trio so those queries look grounded; any other query gets the
// same chunks — it's a stub, clearly labelled.
const cannedChunks = "net/http/server.go:1015: // maxBytesReader caps a request body. The server installs it\n" +
	"  when MaxBytesReader / Server.MaxHeaderBytes apply; reading past the cap\n" +
	"  returns an error and (for the body) sets a 413-style condition.\n\n" +
	"sync/once.go:42: func (o *Once) Do(f func()) { if atomic.LoadUint32(&o.done) == 0 { o.doSlow(f) } }\n" +
	"  // The fast path is a single atomic load; doSlow takes the mutex and\n" +
	"  // re-checks done under it — that double-check is why Once is race-free.\n\n" +
	"runtime/stack.go:1070: // newstack grows a goroutine's stack when a prologue\n" +
	"  stack-check fails: it allocates a larger stack, copies frames, adjusts\n" +
	"  pointers, then reschedules the goroutine on the new stack.\n"

func search(_ context.Context, _ *mcp.CallToolRequest, args searchArgs) (*mcp.CallToolResult, any, error) {
	var b strings.Builder
	b.WriteString("STUB results (not real retrieval) for \"" + args.Query + "\":\n\n")
	b.WriteString(cannedChunks)
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: b.String()}},
	}, nil, nil
}

func main() {
	s := mcp.NewServer(&mcp.Implementation{Name: "ken-stub", Version: "0.1.0"}, nil)
	mcp.AddTool(s, &mcp.Tool{
		Name:        "search",
		Description: "Search the Go standard library source (stub: returns canned file:line chunks).",
	}, search)
	if err := s.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}

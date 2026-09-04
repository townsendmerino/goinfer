package servecheck

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

const usage = `%s check — drive a running goinfer server the way a harness would

  %s check [url]        default http://127.0.0.1:8080

Starts nothing: it exercises whatever is already serving, with the flags its operator chose,
which is the thing Claude Code / dsh / Open-WebUI will actually meet. Every row prints a
NUMBER, because "ok" alone hides the two figures that decide whether a model is usable for a
given harness: time to first token, and the rate after it.

Exits non-zero if any row fails, so it works as a smoke test in a script.
`

// Run implements `serve check`. args excludes the program name and the "check" word.
func Run(args []string, self string) int {
	fs := flag.NewFlagSet("check", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintf(os.Stderr, usage, self, self); fs.PrintDefaults() }
	apiKey := fs.String("api-key", os.Getenv("GOINFER_API_KEY"), "bearer token, if the server was started with -api-key")
	model := fs.String("model", "", "served model id to exercise (default: the first one /v1/models reports)")
	longPrompt := fs.Int("long-prompt", 2000, "word count for the long-prompt TTFT row (0 skips it)")
	timeout := fs.Duration("timeout", 10*time.Minute, "overall deadline")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	base := "http://127.0.0.1:8080"
	if fs.NArg() > 0 {
		base = fs.Arg(0)
	}
	if !strings.Contains(base, "://") {
		base = "http://" + base
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	c := &Client{BaseURL: base, APIKey: *apiKey, Model: *model}

	fmt.Printf("goinfer check → %s\n\n", base)
	var rows []Result

	first, ids := c.Models(ctx)
	rows = append(rows, first)
	target := *model
	if target == "" && len(ids) > 0 {
		target = ids[0]
	}

	if first.OK && target != "" {
		rows = append(rows,
			c.Chat(ctx, target, "Say hello in one short sentence.", 48, "chat, streamed"),
			c.Structured(ctx, target),
			c.Stop(ctx, target),
			c.CountTokens(ctx, target),
		)
		if *longPrompt > 0 {
			// The number a harness user actually needs before choosing a model: TTFT at a
			// realistic turn size, not at a three-word prompt.
			filler := strings.Repeat("the quick brown fox jumps over the lazy dog ", *longPrompt/9+1)
			r := c.Chat(ctx, target, filler+"\n\nIn one word, what animal is mentioned?", 8,
				fmt.Sprintf("long prompt (~%d words)", *longPrompt))
			rows = append(rows, r)
		}
	} else if target == "" {
		rows = append(rows, Result{Name: "chat, streamed", Skip: true, Detail: "no generative model loaded"})
	}

	failed, skipped := 0, 0
	for _, r := range rows {
		status := "ok  "
		switch {
		case r.Skip:
			status = "skip"
			skipped++
		case !r.OK:
			status = "FAIL"
			failed++
		}
		fmt.Printf("%-28s %s  %s\n", r.Name+" "+strings.Repeat(".", max(0, 26-len(r.Name))), status, r.Detail)
	}
	fmt.Println()
	if failed > 0 {
		fmt.Printf("%d of %d checks FAILED\n", failed, len(rows))
		return 1
	}
	// V-17 (docs/review-2026-09-04.md): "all N checks passed" against a server with zero models
	// loaded used to print unconditionally — but with no model, Chat/Structured/Stop/CountTokens
	// never even run (see the else-if above), so N was a shrunk-without-explanation total and the
	// one row that DID run (models list) plus one skip read as full coverage. Same "a SKIP IS NOT
	// A PASS" doctrine this repo already applies to Go test output.
	if skipped > 0 {
		fmt.Printf("%d of %d checks passed (%d skipped, see above)\n",
			len(rows)-skipped-failed, len(rows), skipped)
		return 0
	}
	fmt.Printf("all %d checks passed\n", len(rows))
	return 0
}

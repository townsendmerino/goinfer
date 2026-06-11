// Command stdlib-agent is the terminal REPL over the shared agent core: a
// fully-local RAG coding agent for the Go standard library (goinfer model +
// ken retrieval + aikit underneath). See ../../agent and ../../README.md.
//
// Usage:
//
//	# dev loop (model from disk, ken demo binary on PATH or via --ken):
//	go run ./cmd/stdlib-agent --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	    --ken /path/to/ken-demo-go-stdlib
//
//	# release shape (-tags embed bakes the model in; see build-embed.sh):
//	./stdlib-agent --ken ./ken-demo-go-stdlib
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/demo/agent/agent"
	"github.com/townsendmerino/goinfer/demo/agent/internal/embedmodel"
)

func main() {
	var (
		model   = flag.String("model", "", "path to a .gguf file or HF checkpoint dir (omit in the -tags embed build)")
		ken     = flag.String("ken", "ken-demo-go-stdlib", "path to the ken go-stdlib MCP demo binary (or any ken-mcp server)")
		quant   = flag.String("quant", "int8int8", "weight quant: \"\" | int8 | int8int8 | int4")
		maxTok  = flag.Int("max", 512, "max tokens per answer")
		temp    = flag.Float64("temp", 0.3, "answer-phase sampling temperature (decide phase is always greedy)")
		topK    = flag.Int("top-k", 20, "top-k filter (0 = off)")
		topP    = flag.Float64("top-p", 0.8, "top-p / nucleus (0 = off)")
		kTop    = flag.Int("ken-top-k", 4, "chunks requested per ken search")
		freqPen = flag.Float64("freq-penalty", 0.3, "answer-phase frequency penalty (repetition/loop guard; 0 = off)")
		presPen = flag.Float64("presence-penalty", 0.0, "answer-phase presence penalty (0 = off)")
	)
	flag.Parse()

	opts := agent.Options{
		ModelPath: *model, Quant: *quant,
		KenBin: *ken, KenTopK: *kTop,
		MaxTokens: *maxTok, Temperature: *temp, TopK: *topK, TopP: *topP,
		FrequencyPenalty: *freqPen, PresencePenalty: *presPen,
	}
	if *model == "" {
		raw, ok := embedmodel.Bytes()
		if !ok {
			fmt.Fprintln(os.Stderr, "error: --model is required (or build with -tags embed)")
			flag.Usage()
			os.Exit(2)
		}
		opts.ModelBytes = raw
	}

	progress("loading model…")
	s, err := agent.New(context.Background(), opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		if strings.Contains(err.Error(), "ken") {
			fmt.Fprintln(os.Stderr, "hint: build the ken binary with `go build -tags=kendemo ./demos/go-stdlib` in the ken repo (see its demos/README.md), or pass --ken")
		}
		os.Exit(1)
	}
	defer s.Close()
	progress(s.LoadSummary)
	fmt.Fprintf(os.Stderr, "ken up — tools: %s\n", strings.Join(s.Tools(), ", "))

	repl(s)
}

func repl(s *agent.Session) {
	fmt.Println("\nstdlib-agent — ask the Go standard library anything. /reset clears history, /quit exits.")
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for {
		fmt.Print("\033[1myou>\033[0m ")
		if !in.Scan() {
			fmt.Println("\nbye")
			return
		}
		line := strings.TrimSpace(in.Text())
		switch {
		case line == "":
			continue
		case line == "/quit", line == "/exit":
			fmt.Println("bye")
			return
		case line == "/reset":
			s.Reset()
			fmt.Println("(history cleared)")
			continue
		}
		turn(s, line)
	}
}

// turn runs one exchange, streaming the answer in cyan with dim status lines
// for the agentic machinery. Ctrl-C cancels just this generation.
func turn(s *agent.Session, user string) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	start := time.Now()
	toks := 0
	streaming := false
	_, err := s.Turn(ctx, user, agent.Events{
		Decision: func(action, query string) {
			if action == "search" {
				fmt.Fprintf(os.Stderr, "\033[2m[ken search: %q]\033[0m\n", query)
			}
		},
		Search: func(query, _ string, err error) {
			if err != nil {
				fmt.Fprintf(os.Stderr, "\033[2m[ken error: %v — answering without retrieval]\033[0m\n", err)
			}
		},
		Token: func(text string) {
			if !streaming {
				fmt.Print("\033[36m")
				streaming = true
				start = time.Now()
			}
			toks++ // spans, not tokens — close enough for the rate line
			fmt.Print(text)
		},
	})
	if streaming {
		fmt.Print("\033[0m\n")
	}
	if err != nil && ctx.Err() == nil {
		fmt.Fprintf(os.Stderr, "(error: %v)\n", err)
	}
	if elapsed := time.Since(start); streaming && elapsed > 0 {
		fmt.Fprintf(os.Stderr, "\033[2m[%.1fs]\033[0m\n", elapsed.Seconds())
	}
	_ = toks
}

func progress(m string) { fmt.Fprintln(os.Stderr, m) }

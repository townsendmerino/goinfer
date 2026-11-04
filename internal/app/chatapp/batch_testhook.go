//go:build goinfer_testhooks

package chatapp

import (
	"context"
	"io"
	"os"
	"path/filepath"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/batchio"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// RunBatchForTest runs a batch input file through the --batch runner on a model the caller already loaded, and returns the
// output and error files it wrote. It is the seam internal/serveapp's interchange test uses to put the SAME lines through this
// runner and through serve's POST /v1/batches and compare them: neither side's own tests can show the two agree. Sampling
// defaults are the API's, as with no sampling flags passed. Build-tagged goinfer_testhooks, so it is not in a release binary.
func RunBatchForTest(tk *tokenizer.Tokenizer, model *decoder.Model, name string, think chat.ThinkMode, input []byte) (out, errs []byte, err error) {
	dir, err := os.MkdirTemp("", "batchtest")
	if err != nil {
		return nil, nil, err
	}
	defer os.RemoveAll(dir)
	in, outPath := filepath.Join(dir, "in.jsonl"), filepath.Join(dir, "out.jsonl")
	if err := os.WriteFile(in, input, 0o644); err != nil {
		return nil, nil, err
	}
	plan, err := planBatch(in, outPath)
	if err != nil {
		return nil, nil, err
	}
	s := newSession(tk, model, decoder.Options{}, 0)
	s.think = think
	bd := newBatchDefaults(&chatFlags{}, nil)
	run := func(ctx context.Context, l batchio.Line) lineResult { return s.batchLine(ctx, bd, name, l) }
	if _, err := runBatchLines(context.Background(), plan, run, io.Discard); err != nil {
		return nil, nil, err
	}
	out, _ = os.ReadFile(outPath)
	errs, _ = os.ReadFile(filepath.Join(dir, "out.errors.jsonl"))
	return out, errs, nil
}

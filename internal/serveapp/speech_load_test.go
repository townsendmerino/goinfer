package serveapp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// G-S14i2 of docs/tasks/task-multimodal-support-2026-10.md: a Whisper directory in --model loads as a speech model, beside a decoder when there is one, and a directory missing what it needs is refused by name.

func whisperSmallDir(t *testing.T) string {
	t.Helper()
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "whisper-small")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no whisper-small checkpoint: %v", err)
	}
	return dir
}

func TestNewServer_whisperRefusedByName(t *testing.T) {
	// the tiny fixture has no tokenizer.json: refused at startup, naming the file
	_, err := newServer(config{models: modelFlag{{name: "w", path: filepath.Join("..", "..", "testdata", "whisper-tiny-ts")}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4"}})
	if err == nil || !strings.Contains(err.Error(), "tokenizer.json") {
		t.Fatalf("a Whisper directory without a tokenizer: got %v", err)
	}
}

func TestNewServer_whisperLoadsAsSpeechModel(t *testing.T) {
	dir := whisperSmallDir(t)
	srv, err := newServer(config{models: modelFlag{{name: "w", path: dir}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(srv.models) != 0 || srv.speech["w"] == nil {
		t.Fatalf("models %d, speech %v", len(srv.models), srv.speech)
	}
	if got := srv.servedNames(); len(got) != 1 || got[0] != "w" {
		t.Errorf("servedNames %v", got)
	}
	// a second entry of the same name, speech or decoder, is refused
	_, err = newServer(config{models: modelFlag{{name: "w", path: dir}, {name: "w", path: dir}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4"}})
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Errorf("a duplicate speech name: %v", err)
	}
	// the default name is the directory's
	srv, err = newServer(config{models: modelFlag{{path: dir}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4"}})
	if err != nil || srv.speech["whisper-small"] == nil {
		t.Errorf("default name: %v %v", err, srv.speech)
	}
}

func TestNewServer_whisperBesideADecoder(t *testing.T) {
	dir := whisperSmallDir(t)
	home, _ := os.UserHomeDir()
	dec := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") // no committed decoder fixture has a tokenizer
	if _, err := os.Stat(dec); err != nil {
		t.Skipf("no small decoder checkpoint: %v", err)
	}
	srv, err := newServer(config{models: modelFlag{{name: "d", path: dec}, {name: "w", path: dir}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4"}, kvSessions: 1})
	if err != nil {
		t.Fatal(err)
	}
	if srv.models["d"] == nil || srv.speech["w"] == nil {
		t.Fatalf("models %v speech %v", srv.models, srv.speech)
	}
	if got := strings.Join(srv.servedNames(), ","); got != "d,w" {
		t.Errorf("servedNames %q", got)
	}
}

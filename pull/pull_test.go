package pull

import (
	"strings"
	"testing"
)

func TestParseRef(t *testing.T) {
	for _, tc := range []struct {
		in            string
		repo, file, q string
		wantErr       bool
	}{
		{in: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"},
		{in: "Qwen/X-GGUF:q4_k_m", repo: "Qwen/X-GGUF", q: "q4_k_m"},
		{in: "Qwen/X-GGUF:model-q4_k_m.gguf", repo: "Qwen/X-GGUF", file: "model-q4_k_m.gguf"},
		// .GGUF uppercase still reads as a filename, not a quant.
		{in: "a/b:M.GGUF", repo: "a/b", file: "M.GGUF"},
		{in: "", wantErr: true},
		{in: "noslash", wantErr: true},
		{in: "too/many/slashes", wantErr: true},
		{in: "a/b:", wantErr: true},
		// Path-traversal and URL-injection shapes. "../.." passes a naive
		// "one slash, no leading/trailing slash" check, which is what the first cut used;
		// it would then escape the cache dir under filepath.Join AND corrupt the API URL.
		{in: "../..", wantErr: true},
		{in: "a/../../etc", wantErr: true},
		{in: "./x", wantErr: true},
		{in: "a/..", wantErr: true},
		{in: "../b", wantErr: true},
		{in: "a b/c", wantErr: true},
		{in: "a/b c", wantErr: true},
		{in: "-lead/b", wantErr: true},
		{in: "a/b?x=1", wantErr: true},
		{in: "a/b#frag", wantErr: true},
		{in: "a%2f../b", wantErr: true},
		// ...while ordinary HF names with dots, dashes and underscores stay valid.
		{in: "unsloth/Llama-3.2-3B-Instruct-GGUF", repo: "unsloth/Llama-3.2-3B-Instruct-GGUF"},
		{in: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"},
		{in: "bartowski/google_gemma-3-4b-it-GGUF:Q4_K_M.gguf", repo: "bartowski/google_gemma-3-4b-it-GGUF", file: "Q4_K_M.gguf"},
	} {
		got, err := ParseRef(tc.in)
		if (err != nil) != tc.wantErr {
			t.Fatalf("ParseRef(%q) err = %v, wantErr %v", tc.in, err, tc.wantErr)
		}
		if tc.wantErr {
			continue
		}
		if got.Repo != tc.repo || got.File != tc.file || got.Quant != tc.q {
			t.Errorf("ParseRef(%q) = %+v, want repo=%q file=%q quant=%q", tc.in, got, tc.repo, tc.file, tc.q)
		}
	}
}

// TestSelect_realWorldNaming pins the two naming facts that were MEASURED against the live
// HF API on 2026-09-02, not assumed — the model-pull design doc explicitly refused to commit
// to a matching scheme without checking, and these are what the check found.
//
//  1. Case differs between publishers: Qwen ships "…-q4_k_m.gguf", bartowski ships
//     "…-Q4_K_M.gguf". A case-sensitive match works on one and fails on the other.
//  2. Q2_K and Q2_K_L coexist in the same repo, so a SUBSTRING match on "q2_k" is ambiguous
//     and would silently return whichever sorted first. Matching the full "-<quant>.gguf"
//     suffix is what makes it well-defined.
func TestSelect_realWorldNaming(t *testing.T) {
	qwen := []File{
		{Path: "qwen2.5-coder-1.5b-instruct-q2_k.gguf", Size: 752880192},
		{Path: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", Size: 1117320768, SHA256: "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046"},
		{Path: "qwen2.5-coder-1.5b-instruct-q8_0.gguf", Size: 1894532160},
	}
	bartowski := []File{
		{Path: "google_gemma-3-4b-it-Q2_K.gguf", Size: 1},
		{Path: "google_gemma-3-4b-it-Q2_K_L.gguf", Size: 2},
		{Path: "google_gemma-3-4b-it-Q4_K_M.gguf", Size: 3},
	}

	// lowercase selector against a lowercase repo
	got, err := Select(qwen, Ref{Repo: "Qwen/X", Quant: "q4_k_m"})
	if err != nil {
		t.Fatalf("qwen q4_k_m: %v", err)
	}
	if got.Path != "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" {
		t.Errorf("qwen q4_k_m = %q", got.Path)
	}

	// same lowercase selector must find the UPPERCASE-named file
	got, err = Select(bartowski, Ref{Repo: "b/X", Quant: "q4_k_m"})
	if err != nil {
		t.Fatalf("bartowski q4_k_m: %v", err)
	}
	if got.Path != "google_gemma-3-4b-it-Q4_K_M.gguf" {
		t.Errorf("bartowski q4_k_m = %q", got.Path)
	}

	// q2_k must pick Q2_K exactly, NOT Q2_K_L — the substring trap.
	got, err = Select(bartowski, Ref{Repo: "b/X", Quant: "q2_k"})
	if err != nil {
		t.Fatalf("bartowski q2_k: %v", err)
	}
	if got.Path != "google_gemma-3-4b-it-Q2_K.gguf" {
		t.Errorf("q2_k picked %q, want the exact Q2_K (substring matching would take Q2_K_L)", got.Path)
	}
	// ...and Q2_K_L is still reachable on its own.
	if got, err = Select(bartowski, Ref{Repo: "b/X", Quant: "q2_k_l"}); err != nil || got.Path != "google_gemma-3-4b-it-Q2_K_L.gguf" {
		t.Errorf("q2_k_l = %q, %v", got.Path, err)
	}
}

func TestSelect_errors(t *testing.T) {
	files := []File{{Path: "m-q4_k_m.gguf", Size: 10}, {Path: "m-q8_0.gguf", Size: 20}}

	// A miss must list what IS there — the whole point of the explicit-reference design.
	_, err := Select(files, Ref{Repo: "a/b", Quant: "q5_k_m"})
	if err == nil || !strings.Contains(err.Error(), "m-q8_0.gguf") {
		t.Errorf("missing quant should list available files, got: %v", err)
	}

	// No selector at all is an error that teaches the syntax rather than guessing a default.
	if _, err := Select(files, Ref{Repo: "a/b"}); err == nil || !strings.Contains(err.Error(), "a/b:q4_k_m") {
		t.Errorf("bare repo should suggest a concrete ref, got: %v", err)
	}

	// A split GGUF shard is refused explicitly instead of pulling one useless piece. The
	// selector here is the plain quant a real user would type ("q4_k_m") — not the shard
	// suffix itself — because that IS the bug this used to hide: see
	// TestSelect_splitCheckpoint_namesTheShardsAndOffersAnAlternative below.
	split := []File{{Path: "big-q4_k_m-00001-of-00003.gguf", Size: 1}}
	if _, err := Select(split, Ref{Repo: "a/b", Quant: "q4_k_m"}); err == nil || !strings.Contains(err.Error(), "split GGUF") {
		t.Errorf("split shard should be refused, got: %v", err)
	}

	if _, err := Select(nil, Ref{Repo: "a/b", Quant: "q4_k_m"}); err == nil {
		t.Error("empty repo listing should error")
	}
}

// R15 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the suffix match that decides
// whether a quant selector hits a candidate compared the RAW filename against "-<quant>.gguf" —
// a split shard's filename ends in "-00001-of-00003.gguf", not "-<quant>.gguf", so a real user
// typing the plain quant a split-only repo advertises (":q4_k_m") matched ZERO candidates and
// got "no file matching quant", never reaching the multiPart guard that exists specifically to
// name a split file instead of that generic miss. This is the fixture from the tester's own
// run: a repo whose only q4_k_m is split into 3 parts, plus one single-file alternative quant.
//
// Mutation: revert quantMatchKey to `strings.ToLower(path)` (no shard-suffix stripping, the
// pre-fix behavior) — this test goes red with "no file matching quant" instead of the shard
// error, because the split files never enter cands at all.
func TestSelect_splitCheckpoint_namesTheShardsAndOffersAnAlternative(t *testing.T) {
	files := []File{
		{Path: "big-Q4_K_M-00001-of-00003.gguf", Size: 4 << 30},
		{Path: "big-Q4_K_M-00002-of-00003.gguf", Size: 4 << 30},
		{Path: "big-Q4_K_M-00003-of-00003.gguf", Size: 2 << 30},
		{Path: "big-Q4_K_S.gguf", Size: 8 << 30},
	}
	_, err := Select(files, Ref{Repo: "a/b", Quant: "q4_k_m"})
	if err == nil {
		t.Fatal("split-only quant should be refused, not silently selected")
	}
	if !strings.Contains(err.Error(), "split GGUF") {
		t.Errorf("refusal does not name the split-GGUF case:\n%s", err.Error())
	}
	for _, shard := range []string{"00001-of-00003", "00002-of-00003", "00003-of-00003"} {
		if !strings.Contains(err.Error(), shard) {
			t.Errorf("refusal does not list shard %q:\n%s", shard, err.Error())
		}
	}
	if !strings.Contains(err.Error(), "big-Q4_K_S.gguf") {
		t.Errorf("refusal does not offer the single-file alternative:\n%s", err.Error())
	}

	// No alternative exists: still names the shards, just no "nearest quant" line.
	noAlt := files[:3]
	_, err = Select(noAlt, Ref{Repo: "a/b", Quant: "q4_k_m"})
	if err == nil || !strings.Contains(err.Error(), "split GGUF") {
		t.Fatalf("split-only repo with no alternative should still name the shards, got: %v", err)
	}
	if strings.Contains(err.Error(), "nearest single-file quant") {
		t.Errorf("no single-file quant exists in this fixture, but one was offered:\n%s", err.Error())
	}
}

func TestHumanBytes(t *testing.T) {
	for _, tc := range []struct {
		in   int64
		want string
	}{
		{512, "512 B"}, {1024, "1.0 KiB"}, {1117320768, "1.0 GiB"},
	} {
		if got := HumanBytes(tc.in); got != tc.want {
			t.Errorf("HumanBytes(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

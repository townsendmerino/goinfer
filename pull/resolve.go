package pull

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// IsRef reports whether spec is a model REFERENCE rather than a filesystem path.
//
// Only two prefixes qualify, both unambiguous: "hf:" for a HuggingFace repo and "demo:" for a
// curated tier. Anything else is a path and is returned untouched — a caller's existing
// --model never changes meaning, which is the property that lets this be added to a Hard-tier
// flag at all.
func IsRef(spec string) bool {
	return strings.HasPrefix(spec, "hf:") || strings.HasPrefix(spec, "demo:")
}

// resolveOffline reports an already-cached, verified hit for ref WITHOUT any network access —
// only possible when the exact file, its size, AND its digest are all known statically, which is
// true for a demo: ref (ParseRef sets File+Pin+Bytes from curated.json) and true for NOTHING
// else. A quant selector (hf:repo:quant) genuinely cannot take this path: which file matches the
// quant is not knowable without asking HuggingFace to List the repo. An explicit
// hf:repo:file.gguf ref names its file but was never given a size or digest to verify a cache hit
// against, so it cannot either — trusting a same-sized file on disk without a digest would be
// exactly the "trust the filename" shortcut Download's own cache check deliberately avoids.
//
// V-16 (docs/review-2026-09-04.md): Resolve used to call CheckAccess+List unconditionally before
// any cache check at all, so `serve --model demo:1.5b` with the file already cached still failed
// to start offline — the opposite of what Resolve's own doc comment promised.
func resolveOffline(ref Ref) (string, bool) {
	if ref.File == "" || ref.Pin == "" || ref.Bytes <= 0 {
		return "", false
	}
	dir, err := CacheDir(ref.Repo)
	if err != nil {
		return "", false
	}
	return cachedIntact(dir, File{Path: ref.File, Size: ref.Bytes, SHA256: ref.Pin})
}

// Resolve turns a model spec into a local path, downloading it if necessary.
//
//	Resolve(ctx, "/models/m.gguf")                        → "/models/m.gguf"        (untouched)
//	Resolve(ctx, "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m") → the cached path (pulled if absent)
//	Resolve(ctx, "demo:1.5b")                             → the cached path (pulled if absent)
//
// An already-cached demo: file (exact name, size and digest all pinned in curated.json) is
// returned without touching the network. A quant selector (hf:repo:quant) cannot make the same
// promise: which file matches the quant is not knowable without asking HuggingFace to list the
// repo (V-16, docs/review-2026-09-04.md) — an explicit hf:repo:file.gguf ref names its file but
// still needs List for the size/digest to verify a cache hit against, so it also still checks
// the network first. progress may be nil.
func Resolve(ctx context.Context, spec string, progress func(done, total int64)) (string, error) {
	if !IsRef(spec) {
		return spec, nil
	}
	ref, err := ParseRef(strings.TrimPrefix(spec, "hf:"))
	if err != nil {
		return "", err
	}
	if ref.File == "" && ref.Quant == "" {
		return "", fmt.Errorf("%s: name a quant or a file, e.g. hf:%s:q4_k_m", spec, ref.Repo)
	}
	// V-16 (docs/review-2026-09-04.md): checked BEFORE any network call. See resolveOffline.
	if path, ok := resolveOffline(ref); ok {
		return path, nil
	}
	if err := CheckAccess(ctx, ref.Repo); err != nil {
		return "", err
	}
	files, err := List(ctx, ref.Repo)
	if err != nil {
		return "", err
	}
	f, err := Select(files, ref)
	if err != nil {
		return "", err
	}
	dir, err := CacheDir(ref.Repo)
	if err != nil {
		return "", err
	}
	return Download(ctx, ref.Repo, f, dir, progress)
}

// ResolveVerbose is Resolve with a progress line on stderr, for the command-line front ends.
// Split from Resolve so a library caller gets no surprise output on a stream it does not own.
func ResolveVerbose(ctx context.Context, spec string) (string, error) {
	if !IsRef(spec) {
		return spec, nil
	}
	fmt.Fprintf(os.Stderr, "resolving %s\n", spec)
	path, err := Resolve(ctx, spec, func(done, total int64) {
		if total > 0 {
			fmt.Fprintf(os.Stderr, "\r  %s / %s (%d%%)   ", HumanBytes(done), HumanBytes(total), 100*done/total)
		}
	})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return "", err
	}
	fmt.Fprintf(os.Stderr, "  %s\n", path)
	return path, nil
}

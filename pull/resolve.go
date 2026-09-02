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

// Resolve turns a model spec into a local path, downloading it if necessary.
//
//	Resolve(ctx, "/models/m.gguf")                        → "/models/m.gguf"        (untouched)
//	Resolve(ctx, "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m") → the cached path (pulled if absent)
//	Resolve(ctx, "demo:1.5b")                             → the cached path (pulled if absent)
//
// An already-cached file with the right size and digest is returned without touching the
// network. progress may be nil.
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

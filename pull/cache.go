package pull

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// CacheEntry is one model the pull cache holds (task-checkpoint-fetch-2026-09.md P8): what is ON DISK, which is a
// different question from what a server has loaded.
type CacheEntry struct {
	Repo string // owner/repo, from the cache layout (<root>/<owner>/<repo>)
	// Path is what --model, or the web UI's load, takes: the .gguf, a split set's first shard, or the checkpoint
	// directory. For an incomplete entry it is where that file or directory will be once the pull finishes.
	Path     string
	Kind     string // "gguf", "split" or "checkpoint"
	Bytes    int64  // on disk now: the file, every shard present, or the whole directory (its staging copy while incomplete)
	Shards   int    // a split set's shard count; 0 otherwise
	Complete bool   // false for an interrupted pull: a .part file, a split set missing a shard, a checkpoint still in staging
	// Sidecars is the bytes of the .giw sidecars built from this model, beside it: the transcoded copy a load runs from,
	// which costs disk on top of the download.
	Sidecars int64
}

// CacheEntries lists what the pull cache holds, sorted by repo then path. It reads the cache layout and file sizes
// only: a checkpoint counts as complete when its marker is there and every file it names has its recorded size, and no
// digest is computed, so listing a cache of large models stays fast. Loading still verifies in full. A missing cache is
// an empty list, not an error.
func CacheEntries() ([]CacheEntry, error) {
	root, err := CacheRoot()
	if err != nil {
		return nil, err
	}
	owners, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []CacheEntry
	for _, o := range owners {
		if !o.IsDir() {
			continue
		}
		repos, err := os.ReadDir(filepath.Join(root, o.Name()))
		if err != nil {
			continue
		}
		for _, r := range repos {
			if !r.IsDir() {
				continue
			}
			dir := filepath.Join(root, o.Name(), r.Name())
			if name, ok := strings.CutSuffix(r.Name(), ".partial"); ok {
				// A checkpoint still being assembled (DownloadCheckpoint's staging directory).
				out = append(out, CacheEntry{Repo: o.Name() + "/" + name, Path: filepath.Join(root, o.Name(), name), Kind: "checkpoint", Bytes: dirBytes(dir)})
				continue
			}
			out = append(out, repoEntries(o.Name()+"/"+r.Name(), dir)...)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Repo != out[j].Repo {
			return out[i].Repo < out[j].Repo
		}
		return out[i].Path < out[j].Path
	})
	return out, nil
}

// repoEntries is one repo directory's models: a checkpoint (the directory itself, when it carries the marker), and the
// GGUF files in it, a split set folded into one entry, with each model's .giw sidecars counted against it.
func repoEntries(repo, dir string) []CacheEntry {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []CacheEntry
	inMarker := map[string]bool{}
	if b, err := os.ReadFile(filepath.Join(dir, checkpointMarker)); err == nil {
		var m markerDoc
		if json.Unmarshal(b, &m) == nil && len(m.Files) > 0 {
			complete := true
			for _, f := range m.Files {
				inMarker[filepath.Base(f.Path)] = true
				if st, err := os.Stat(filepath.Join(dir, filepath.FromSlash(f.Path))); err != nil || st.Size() != f.Size {
					complete = false
				}
			}
			out = append(out, CacheEntry{Repo: repo, Path: dir, Kind: "checkpoint", Bytes: dirBytes(dir), Complete: complete})
		}
	}
	sizes := map[string]int64{}
	var giws []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		name := e.Name()
		sizes[name] = info.Size()
		if strings.HasSuffix(name, ".giw") {
			giws = append(giws, name)
		}
	}
	sidecarsFor := func(gguf string) int64 {
		stem := strings.TrimSuffix(gguf, filepath.Ext(gguf)) + "."
		var n int64
		for _, g := range giws {
			if strings.HasPrefix(g, stem) {
				n += sizes[g]
			}
		}
		return n
	}
	sets := map[string][]string{} // a split set's "<prefix>-of-<NNNNN>" -> the shard names present
	for name, size := range sizes {
		if inMarker[name] {
			continue // part of the checkpoint entry above
		}
		lower := strings.ToLower(name)
		switch {
		case strings.HasSuffix(lower, ".gguf.part"):
			final := strings.TrimSuffix(name, ".part")
			if _, done := sizes[final]; !done {
				out = append(out, CacheEntry{Repo: repo, Path: filepath.Join(dir, final), Kind: "gguf", Bytes: size})
			}
		case strings.HasSuffix(lower, ".gguf"):
			if m := shardSuffix.FindStringSubmatch(lower); m != nil {
				key := name[:len(m[1])] + "-of-" + m[3]
				sets[key] = append(sets[key], name)
				continue
			}
			out = append(out, CacheEntry{Repo: repo, Path: filepath.Join(dir, name), Kind: "gguf", Bytes: size, Complete: true, Sidecars: sidecarsFor(name)})
		}
	}
	for _, shards := range sets {
		sort.Strings(shards)
		m := shardSuffix.FindStringSubmatch(strings.ToLower(shards[0]))
		n, _ := strconv.Atoi(m[3]) // five digits, matched by shardSuffix
		var bytes int64
		for _, s := range shards {
			bytes += sizes[s]
		}
		first := shards[0][:len(m[1])] + "-00001-of-" + m[3] + filepath.Ext(shards[0])
		e := CacheEntry{Repo: repo, Path: filepath.Join(dir, first), Kind: "split", Bytes: bytes, Shards: n, Complete: len(shards) == n && shards[0] == first}
		if e.Complete {
			e.Sidecars = sidecarsFor(first)
		}
		out = append(out, e)
	}
	return out
}

// dirBytes is the size of every regular file under dir.
func dirBytes(dir string) int64 {
	var n int64
	_ = filepath.WalkDir(dir, func(_ string, d os.DirEntry, err error) error {
		if err == nil && d.Type().IsRegular() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}

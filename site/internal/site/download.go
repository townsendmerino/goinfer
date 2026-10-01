package site

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// The Download page (docs/tasks/task-site-2026-09.md S4) is built from the release facts scripts/site_release_data.py writes
// to site/data/release.json: the tag, and every attached asset with its size and the sha256 from the release's own
// checksums.txt. The file is generated at build time and not committed; without it (a local build with no network) the page
// says so and points at the releases page rather than showing a table it cannot fill.

// ReleaseAsset is one file attached to the release.
type ReleaseAsset struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
	URL  string `json:"url"`
}

// Release is site/data/release.json.
type Release struct {
	Tag       string            `json:"tag"`
	Published string            `json:"published"`
	URL       string            `json:"url"`
	Assets    []ReleaseAsset    `json:"assets"`
	Checksums map[string]string `json:"checksums"`
}

// DLFile is one downloadable binary.
type DLFile struct{ Name, URL, SHA, Size string }

// DLPlatform is a column of the download grid.
type DLPlatform struct{ Key, Label string }

// DLKind is one kind of binary, with its file for each platform that has one.
type DLKind struct {
	Key, Title, What string
	Files            map[string]*DLFile // by platform key
}

// Download is what the page shows.
type Download struct {
	Rel       *Release
	Platforms []DLPlatform
	Kinds     []DLKind
	Others    []DLFile // checksums, notice, licence
	Sample    *DLFile  // one real file, for the example commands
}

var dlPlatforms = []DLPlatform{
	{"darwin-arm64", "macOS · Apple silicon"}, {"darwin-amd64", "macOS · Intel"},
	{"linux-amd64", "Linux · x86-64"}, {"linux-arm64", "Linux · ARM64"},
	{"windows-amd64", "Windows · x86-64"}, {"windows-arm64", "Windows · ARM64"},
}

// The what-it-is lines are the README's own table (README.md, "Binaries on the latest release").
var dlKinds = []DLKind{
	{Key: "goinfer-serve", Title: "Server", What: "The server: OpenAI and Anthropic APIs, the web UI, GPU support built in."},
	{Key: "goinfer-chat", Title: "Chat runtime", What: "The single-shot runtime. Point it at your own GGUF."},
	{Key: "goinfer-chat-0.5b", Title: "Chat with a 0.5B model inside", What: "Runtime and model in one file. Nothing to install or download."},
	{Key: "goinfer-chat-1.5b", Title: "Chat with a 1.5B model inside", What: "The same, with the 1.5B coder model."},
}

var assetRE = regexp.MustCompile(`^(goinfer-serve|goinfer-chat-0\.5b|goinfer-chat-1\.5b|goinfer-chat)-((?:darwin|linux|windows)-(?:amd64|arm64))(?:\.exe)?$`)

// LoadRelease reads site/data/release.json; a missing file is not an error (nil).
func LoadRelease(root string) (*Release, error) {
	b, err := os.ReadFile(filepath.Join(root, "site", "data", "release.json"))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var r Release
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("site/data/release.json: %w", err)
	}
	if r.Tag == "" || len(r.Assets) == 0 {
		return nil, fmt.Errorf("site/data/release.json has no tag or no assets")
	}
	return &r, nil
}

func humanSize(n int64) string {
	switch {
	case n >= 1<<30:
		return strconv.FormatFloat(float64(n)/(1<<30), 'f', 2, 64) + " GB"
	case n >= 1<<20:
		return strconv.FormatFloat(float64(n)/(1<<20), 'f', 0, 64) + " MB"
	}
	return strconv.FormatFloat(float64(n)/(1<<10), 'f', 0, 64) + " KB"
}

// BuildDownload arranges the release's assets into the grid, and refuses a release whose binaries do not add up.
func BuildDownload(r *Release) (*Download, error) {
	if r == nil {
		return nil, nil
	}
	d := &Download{Rel: r, Platforms: dlPlatforms}
	d.Kinds = make([]DLKind, len(dlKinds))
	copy(d.Kinds, dlKinds)
	for i := range d.Kinds {
		d.Kinds[i].Files = map[string]*DLFile{}
	}
	for _, a := range r.Assets {
		f := &DLFile{Name: a.Name, URL: a.URL, SHA: r.Checksums[a.Name], Size: humanSize(a.Size)}
		m := assetRE.FindStringSubmatch(a.Name)
		if m == nil {
			d.Others = append(d.Others, *f)
			continue
		}
		if f.SHA == "" {
			return nil, fmt.Errorf("release %s: %s has no checksum in checksums.txt", r.Tag, a.Name)
		}
		for i := range d.Kinds {
			if d.Kinds[i].Key == m[1] {
				d.Kinds[i].Files[m[2]] = f
			}
		}
		// The example commands use the model-included 0.5B, for Apple silicon when the release has it: the page cannot
		// tell the visitor's arch (download.js), and darwin-arm64 is what the README's own examples use.
		if m[1] == "goinfer-chat-0.5b" && (d.Sample == nil || m[2] == "darwin-arm64") {
			d.Sample = f
		}
	}
	for _, k := range d.Kinds {
		if len(k.Files) == 0 {
			return nil, fmt.Errorf("release %s carries no %s binary", r.Tag, k.Key)
		}
	}
	if d.Sample == nil {
		for _, k := range d.Kinds {
			for _, f := range k.Files {
				d.Sample = f
			}
		}
	}
	return d, nil
}

// Shell is a file name safe to show in a command line.
func (f *DLFile) Shell() string { return strings.TrimSuffix(f.Name, ".exe") }

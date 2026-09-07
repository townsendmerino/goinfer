package pull

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// The recommended-checkpoint registry: a short name a person can type, mapped to a checkpoint this
// project has actually run.
//
// THE PROBLEM IT SOLVES. `pull owner/repo:quant` works and requires the user to already know three
// things — that a GGUF conversion exists, who published it, and which quantization to ask for.
// That is knowledge from having spent time on Hugging Face, which is exactly what a first-time
// user does not have.
//
// IT DERIVES FROM THE CAPABILITY MATRIX, and that is the whole design. docs/capability-matrix.json
// already records which families this project supports and at what parity status; a hand-kept
// second list would drift from it, and a registry claiming support the matrix does not back is
// worse than no registry. So a checkpoint entry lives ON its family's matrix row, and
// TestRegistry_everyEntryTracesToItsFamily fails if one ever names a family that is not there.
//
// NOT A DOWNLOAD SERVICE. Every entry points at Hugging Face and carries a sha256 the fetch
// verifies. This project hosts no weights: it costs money, creates an availability obligation
// nobody here can meet, and redistribution carries licence questions worth avoiding.
//
// DISTINCT FROM `demo:` TIERS, deliberately. curated.json pins the models that are EMBEDDED in
// release binaries and is checked against the release workflow; its own comment says it is "not a
// name registry to grow". That is a different question from "which checkpoints do we recommend",
// so this does not extend it.

//go:embed capability-matrix.json
var capabilityMatrixJSON []byte

// Checkpoint is one recommended checkpoint, as recorded on its capability-matrix row.
type Checkpoint struct {
	Name    string `json:"name"`     // the short name `pull` accepts
	Repo    string `json:"repo"`     // Hugging Face repo
	File    string `json:"file"`     // exact filename in that repo
	Quant   string `json:"quant"`    // the on-disk quantization
	Bytes   int64  `json:"bytes"`    // download size
	SHA256  string `json:"sha256"`   // verified on fetch
	GoodFor string `json:"good_for"` // what it is worth using for
	Needs   string `json:"needs"`    // what it costs to run
	// Tools records what `internal/servecheck`'s two tools rows measured for this checkpoint
	// (R11, docs/measurements/cold-user-2026-09-06-nobara-pc.md): "tools, OpenAI" is a
	// one-function schema, "tools, harness-scale" a dozen-tool schema shaped like a real agent's
	// — the shape that broke under opencode with a server whose minimal-schema row was green.
	// From a RECORDED `serve check` run, never guessed (TestRegistry_toolsColumnIsNonEmpty).
	Tools string `json:"tools"`

	// Family and Parity are copied off the matrix row at load, so a caller listing checkpoints can
	// show what backs each one without re-reading the matrix.
	Family string `json:"-"`
	Parity string `json:"-"`
}

type matrixRow struct {
	Name       string      `json:"name"`
	Parity     string      `json:"parity"`
	Checkpoint *Checkpoint `json:"checkpoint"`
}

var registry = struct {
	once   sync.Once
	byName map[string]Checkpoint
}{}

func loadRegistry() map[string]Checkpoint {
	registry.once.Do(func() {
		var rows []matrixRow
		if err := json.Unmarshal(capabilityMatrixJSON, &rows); err != nil {
			// A build-time asset: a parse failure is a bug in the tree, not bad input.
			panic("pull: capability-matrix.json is malformed: " + err.Error())
		}
		registry.byName = map[string]Checkpoint{}
		for _, r := range rows {
			if r.Checkpoint == nil {
				continue
			}
			c := *r.Checkpoint
			c.Family, c.Parity = r.Name, r.Parity
			registry.byName[c.Name] = c
		}
	})
	return registry.byName
}

// Recommended looks up a short name. ok is false for anything not in the registry, which the
// caller should treat as "maybe it is an owner/repo ref" rather than as an error.
func Recommended(name string) (Checkpoint, bool) {
	c, ok := loadRegistry()[strings.ToLower(strings.TrimSpace(name))]
	return c, ok
}

// RecommendedAll returns every entry, sorted by download size so the cheapest thing to try is
// first — which is the order a first-time user wants and the reverse of alphabetical.
func RecommendedAll() []Checkpoint {
	m := loadRegistry()
	out := make([]Checkpoint, 0, len(m))
	for _, c := range m {
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Bytes < out[j].Bytes })
	return out
}

// RecommendedNames returns the short names, size order, for help text and error messages.
func RecommendedNames() []string {
	all := RecommendedAll()
	names := make([]string, len(all))
	for i, c := range all {
		names[i] = c.Name
	}
	return names
}

// Ref converts a registry entry into the exact repo/file reference the fetcher already takes, so
// a short name is a lookup in front of the existing path rather than a second download route.
func (c Checkpoint) Ref() string { return c.Repo + ":" + c.File }

// Describe is one listing line.
func (c Checkpoint) Describe() string {
	return fmt.Sprintf("%-22s %6.2f GB  %-8s %s", c.Name, float64(c.Bytes)/1e9, c.Quant, c.GoodFor)
}

// DescribeTools reports what `serve check`'s two tools rows measured for this checkpoint (R11):
// whether it calls a tool at all, and separately, whether it still does under a real agent's
// full tool schema — the two are not the same fact, and a model can pass the first and skip the
// second.
func (c Checkpoint) DescribeTools() string {
	if c.Tools == "" {
		return "not yet measured"
	}
	return c.Tools
}

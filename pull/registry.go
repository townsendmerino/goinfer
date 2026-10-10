package pull

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// The recommended-checkpoint registry: a short name a person can type (`pull <name>`), mapped to a
// checkpoint this project has actually run, so a first-time user does not need to know that a GGUF
// conversion exists, who published it, or which quantization to ask for.
//
// IT DERIVES FROM THE CAPABILITY MATRIX. docs/capability-matrix.json records which families this project
// supports and at what parity status; a hand-kept second list would drift from it, and a registry
// claiming support the matrix does not back is worse than no registry. So a checkpoint entry lives ON
// its family's matrix row, and TestRegistry_everyEntryTracesToItsFamily fails if one names a family that
// is not there. pull/capability-matrix.json is the embedded byte copy of that file (the docs file is
// canonical; a test fails on drift).
//
// NOT A DOWNLOAD SERVICE. Every entry points at Hugging Face and carries a sha256 the fetch verifies.
// This project hosts no weights: that creates an availability obligation nobody here can meet, and
// redistribution carries licence questions worth avoiding.
//
// DISTINCT FROM `demo:` TIERS, deliberately. curated.json pins the models embedded in release binaries,
// is checked against the release workflow, and is "not a name registry to grow", so this does not extend
// it.

//go:embed capability-matrix.json
var capabilityMatrixJSON []byte

// Checkpoint is one recommended checkpoint, as recorded on its capability-matrix row.
type Checkpoint struct {
	Name   string `json:"name"`   // the short name `pull` accepts
	Repo   string `json:"repo"`   // Hugging Face repo
	File   string `json:"file"`   // exact filename in that repo ("" for a directory entry)
	Quant  string `json:"quant"`  // the on-disk quantization
	Bytes  int64  `json:"bytes"`  // download size
	SHA256 string `json:"sha256"` // verified on fetch (for a directory entry: the plan's tree digest, Plan.TreeDigest)
	// Kind is "" for a single-file GGUF and KindDirectory for a safetensors checkpoint fetched as a set
	// (`pull <name>` then runs the same plan-and-verify path as `pull owner/repo:safetensors`, and refuses
	// when the repo's files no longer match the tree digest this build pins). It is how a vision-language
	// checkpoint can be recommended at all: the GGUF loader reads no image projector, the safetensors
	// directory carries the tower (docs/multimodal.md).
	Kind    string `json:"kind,omitempty"`
	GoodFor string `json:"good_for"` // what it is worth using for
	Needs   string `json:"needs"`    // what it costs to run
	// Tools records what `internal/servecheck`'s two tools rows measured for this checkpoint: "tools,
	// OpenAI" is a one-function schema, "tools, harness-scale" a dozen-tool schema shaped like a real
	// agent's. From a RECORDED `serve check` run, never guessed (TestRegistry_toolsColumnIsNonEmpty).
	Tools string `json:"tools"`

	// Family and Parity are copied off the matrix row at load, so a caller listing checkpoints can
	// show what backs each one without re-reading the matrix.
	Family string `json:"-"`
	Parity string `json:"-"`
}

// KindDirectory marks a registry entry that is a safetensors checkpoint directory, not a single GGUF file.
const KindDirectory = "directory"

// IsDirectory reports whether the entry is a safetensors checkpoint directory.
func (c Checkpoint) IsDirectory() bool { return c.Kind == KindDirectory }

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
func (c Checkpoint) Ref() string {
	if c.IsDirectory() {
		return c.Repo + ":" + CheckpointSelector
	}
	return c.Repo + ":" + c.File
}

// Describe is one listing line.
func (c Checkpoint) Describe() string {
	quant := c.Quant
	if c.IsDirectory() {
		quant += " dir" // a safetensors directory, quantized on load: the size is the full-precision download
	}
	return fmt.Sprintf("%-22s %6.2f GB  %-8s %s", c.Name, float64(c.Bytes)/1e9, quant, c.GoodFor)
}

// DescribeTools reports what `serve check`'s two tools rows measured for this checkpoint: whether it
// calls a tool at all, and separately whether it still does under a real agent's full tool schema.
// The two are not the same fact; a model can pass the first and skip the second.
func (c Checkpoint) DescribeTools() string {
	if c.Tools == "" {
		return "not yet measured"
	}
	return c.Tools
}

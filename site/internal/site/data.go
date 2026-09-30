// Package site builds goinfer.dev from the repository (docs/tasks/task-site-2026-09.md, S8a). It reads the generated
// capability matrix, the curated release tiers and the site's own two small data files, and writes plain static files.
// Nothing here talks to the network, and nothing is hand-listed that the repo can say for itself: a new family, a new
// checkpoint or a new curated tier appears on the next build.
package site

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Row is one family's row in docs/capability-matrix.json.
type Row struct {
	Name          string         `json:"name"`
	ModelTypes    []string       `json:"model_types"`
	DisplayName   string         `json:"display_name"`
	Description   string         `json:"description"`
	Summary       string         `json:"summary"`
	Tasks         []string       `json:"tasks"`
	CoverageAxis  string         `json:"coverage_axis"`
	MoE           string         `json:"moe"`
	SlidingWindow string         `json:"sliding_window"`
	QKNorm        bool           `json:"qk_norm"`
	RopeStyle     string         `json:"rope_style"`
	Norm          string         `json:"norm"`
	Activation    string         `json:"activation"`
	TiedLMHead    bool           `json:"tied_lm_head"`
	Loaders       string         `json:"loaders"`
	Modality      string         `json:"modality"`
	GPUResident   bool           `json:"gpu_residency_eligible"`
	Parity        string         `json:"parity"`
	Checkpoint    *RegCheckpoint `json:"checkpoint"`
}

// RegCheckpoint is the recommended checkpoint on a family's row (the registry pull reads).
type RegCheckpoint struct {
	Name    string `json:"name"`
	Label   string `json:"label"`
	Repo    string `json:"repo"`
	File    string `json:"file"`
	Quant   string `json:"quant"`
	Bytes   int64  `json:"bytes"`
	SHA256  string `json:"sha256"`
	GoodFor string `json:"good_for"`
	Needs   string `json:"needs"`
	Tools   string `json:"tools"`
}

// Curated is pull/curated.json: the tiers goinfer ships embedded in its release binaries.
type Curated struct {
	Tiers map[string]struct {
		Repo   string `json:"repo"`
		File   string `json:"file"`
		SHA256 string `json:"sha256"`
		Bytes  int64  `json:"bytes"`
	} `json:"tiers"`
}

// Machine is one measured machine (site/data/machines.json).
type Machine struct {
	Key      string  `json:"key"`
	Name     string  `json:"name"`
	Short    string  `json:"short"`
	Label    string  `json:"label"`
	Path     string  `json:"path"`
	MemoryGB float64 `json:"memory_gb"`
	GPU      string  `json:"gpu"`
}

// Claim is one measured figure the site shows (site/data/claims.json), with where it came from.
type Claim struct {
	ID         string   `json:"id"`
	Checkpoint string   `json:"checkpoint"`
	Machine    string   `json:"machine"`
	Value      *float64 `json:"value"`
	Peer       *float64 `json:"peer"`
	Basis      string   `json:"basis"`     // median-of-3 | mean | none: what Value and Peer are
	Runs       string   `json:"runs"`      // basis median-of-3: the three runs exactly as the record prints them
	PeerRuns   string   `json:"peer_runs"` // the same, for the peer
	RatioRaw   string   `json:"ratio_raw"` // the record's own median ratio, as printed there
	Ratio      string   `json:"ratio"`     // what the page shows: RatioRaw rounded to 2 places, or a plain reason
	Verdict    string   `json:"verdict"`   // ahead | behind | na
	Why        string   `json:"why"`
	Date       string   `json:"date"`
	Source     Source   `json:"source"`
}

// Fact is a sentence the site states that a record must also state.
type Fact struct {
	Text   string `json:"text"`
	Source Source `json:"source"`
}

// Source names the file, and the heading in it, a claim's figures must appear under.
type Source struct {
	Path    string `json:"path"`
	Heading string `json:"heading"`
}

// DecisionClaim is one checkpoint's measured decision figures (the S2 decisions row, D9 of
// docs/tasks/task-constrained-confidence.md): label scoring (Route A) through POST /v1/systemone, top-1 and ECE on a
// labelled sample, both of which must appear in the cited record, as a speed must.
type DecisionClaim struct {
	ID         string `json:"id"`
	Checkpoint string `json:"checkpoint"`
	Machine    string `json:"machine"`
	Route      string `json:"route"`    // label (Route A); head once a checkpoint has a trained decision head
	Template   string `json:"template"` // chat-v1 or bare-v1
	Top1       string `json:"top1"`     // as the record prints it
	ECE        string `json:"ece"`
	Calibrated bool   `json:"calibrated"`
	Sample     string `json:"sample"` // what the figures are over, in words
	Date       string `json:"date"`
	Source     Source `json:"source"`
}

// Claims is the whole claims file.
type Claims struct {
	Method    string          `json:"method"`
	Facts     []Fact          `json:"facts"`
	Claims    []Claim         `json:"claims"`
	Decisions []DecisionClaim `json:"decisions"`
}

// Inputs is everything a build reads.
type Inputs struct {
	Rows     []Row
	Curated  Curated
	Machines []Machine
	Claims   Claims
	Ollama   *OllamaData // site/data/ollama.json, the "Coming from Ollama?" section
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// LoadInputs reads the repo at root. The matrix is docs/capability-matrix.json, the canonical file: pull/ embeds a byte
// copy of it that a test keeps in lockstep.
func LoadInputs(root string) (*Inputs, error) {
	in := &Inputs{}
	if err := readJSON(filepath.Join(root, "docs", "capability-matrix.json"), &in.Rows); err != nil {
		return nil, err
	}
	if err := readJSON(filepath.Join(root, "pull", "curated.json"), &in.Curated); err != nil {
		return nil, err
	}
	var m struct {
		Machines []Machine `json:"machines"`
	}
	if err := readJSON(filepath.Join(root, "site", "data", "machines.json"), &m); err != nil {
		return nil, err
	}
	in.Machines = m.Machines
	if err := readJSON(filepath.Join(root, "site", "data", "claims.json"), &in.Claims); err != nil {
		return nil, err
	}
	var err error
	if in.Ollama, err = LoadOllama(root); err != nil {
		return nil, err
	}
	return in, nil
}

var slugRe = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)

// Validate refuses inputs the pages could not be built from, naming the first thing wrong. It is the generator's own
// gate on the data; the claims check (claims.go) is the gate on the figures.
func (in *Inputs) Validate() error {
	if len(in.Rows) == 0 {
		return fmt.Errorf("the capability matrix has no families")
	}
	seen := map[string]bool{}
	for _, r := range in.Rows {
		switch {
		case !slugRe.MatchString(r.Name):
			return fmt.Errorf("family %q is not a usable page name", r.Name)
		case seen[r.Name]:
			return fmt.Errorf("family %q appears twice", r.Name)
		case strings.TrimSpace(r.DisplayName) == "" || strings.TrimSpace(r.Summary) == "" || len(r.Tasks) == 0:
			return fmt.Errorf("family %q is missing its display name, summary or tasks (the registry's siteDocs)", r.Name)
		case strings.TrimSpace(r.Parity) == "":
			return fmt.Errorf("family %q has no parity string", r.Name)
		}
		seen[r.Name] = true
	}
	mk := map[string]bool{}
	for _, m := range in.Machines {
		if m.Key == "" || m.MemoryGB <= 0 || m.Name == "" {
			return fmt.Errorf("machine %+v is incomplete", m)
		}
		mk[m.Key] = true
	}
	for _, c := range in.Claims.Claims {
		if !mk[c.Machine] {
			return fmt.Errorf("claim %q names machine %q, which machines.json does not list", c.ID, c.Machine)
		}
		if c.Date == "" {
			return fmt.Errorf("claim %q has no date: every number names its machine and date", c.ID)
		}
		if c.Source.Path == "" || c.Source.Heading == "" {
			return fmt.Errorf("claim %q has no source", c.ID)
		}
	}
	for _, d := range in.Claims.Decisions {
		switch {
		case !mk[d.Machine]:
			return fmt.Errorf("decision claim %q names machine %q, which machines.json does not list", d.ID, d.Machine)
		case d.Date == "" || d.Top1 == "" || d.ECE == "" || d.Sample == "":
			return fmt.Errorf("decision claim %q needs top1, ece, sample and date", d.ID)
		case d.Route != "label" && d.Route != "head":
			return fmt.Errorf("decision claim %q: route %q must be label or head", d.ID, d.Route)
		case d.Source.Path == "" || d.Source.Heading == "":
			return fmt.Errorf("decision claim %q has no source", d.ID)
		}
	}
	return nil
}

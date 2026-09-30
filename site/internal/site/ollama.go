package site

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// The "Coming from Ollama?" section of the Models page (docs/tasks/task-site-2026-09.md S2): Ollama's most-pulled
// models, each with what goinfer does with it. site/data/ollama.json transcribes a dated snapshot in docs/measurements/
// row for row; the status is not stored but derived from the capability matrix at build time; CheckOllama holds the
// data, the derived statuses and the headline to the snapshot.

// OllamaRow is one row of site/data/ollama.json.
type OllamaRow struct {
	Rank       int      `json:"rank"`
	Tag        string   `json:"tag"`
	PullsM     float64  `json:"pulls_m"`
	Caps       []string `json:"ollama_caps"`
	Families   []string `json:"families"` // matrix family names, or "encoder:<kind>" for an aikit encoder
	Note       string   `json:"note"`
	Unverified bool     `json:"unverified"`
	// Needs is, for a not-supported row, the Hugging Face model_type the tag's weights use, so the build can tell when
	// the capability matrix gains it: a row with no family can otherwise never change status, and support landing for
	// it would leave the page saying "Not supported yet" (it did, for Gemma 1 and 2, before this field existed).
	Needs string `json:"needs"`
}

// OllamaData is site/data/ollama.json.
type OllamaData struct {
	Snapshot string      `json:"snapshot"` // the measurement file the rows transcribe
	Read     string      `json:"read"`     // the date the library page was read
	Headline string      `json:"headline"` // the snapshot's sentence; also a Fact in claims.json
	Rows     []OllamaRow `json:"rows"`
}

// encoderKinds are the embedders aikit's encoder loads (goinfer serve --embed-model), which the matrix does not list
// because they are not decoder families. A row naming one is supported by definition.
var encoderKinds = map[string]bool{"bert": true, "nomic-bert": true, "xlm-roberta": true, "minilm": true, "arctic": true}

// The statuses, as the snapshot writes them.
const (
	OllamaSupported   = "S"
	OllamaTextOnly    = "T"
	OllamaUnsupported = "N"
	OllamaUnverified  = "U"
)

// OllamaStatusText is what the page says for each status.
var OllamaStatusText = map[string]string{
	OllamaSupported:   "Runs",
	OllamaTextOnly:    "Runs as text only",
	OllamaUnsupported: "Not supported yet",
	OllamaUnverified:  "Not checked yet",
}

// OllamaEntry is a row with what the build derives for it.
type OllamaEntry struct {
	OllamaRow
	Status  string
	Links   []*Family // the families with a page
	Encoder bool      // an encoder row, which links the Embeddings section
}

// StatusText is the status in words.
func (e *OllamaEntry) StatusText() string { return OllamaStatusText[e.Status] }

// Ollama is the section's model: the data and its derived entries.
type Ollama struct {
	*OllamaData
	Entries []*OllamaEntry
}

// Top is the first rows the section shows open; Rest is behind a disclosure.
func (o *Ollama) Top() []*OllamaEntry  { return o.Entries[:min(ollamaOpenRows, len(o.Entries))] }
func (o *Ollama) Rest() []*OllamaEntry { return o.Entries[min(ollamaOpenRows, len(o.Entries)):] }

const ollamaOpenRows = 20

// LoadOllama reads site/data/ollama.json.
func LoadOllama(root string) (*OllamaData, error) {
	var d OllamaData
	if err := readJSON(filepath.Join(root, "site", "data", "ollama.json"), &d); err != nil {
		return nil, err
	}
	if d.Snapshot == "" || d.Read == "" || d.Headline == "" || len(d.Rows) == 0 {
		return nil, fmt.Errorf("site/data/ollama.json: snapshot, read, headline and rows are all required")
	}
	return &d, nil
}

// hasVision reports whether goinfer runs a family's images. The matrix's tasks list carries "vision" for some families
// and not for gemma3, whose images are recorded only in modality ("text (+ vision via VL text_config)"); modality
// mentions vision for every family that has it, and says "ignored" where a checkpoint's tower is skipped (mistral3).
func hasVision(r Row) bool {
	return slices.Contains(r.Tasks, "vision") || (strings.Contains(r.Modality, "vision") && !strings.Contains(r.Modality, "ignored"))
}

// deriveOllama gives each row its status from the matrix:
//   - N: no family, or a family the matrix does not have (an unknown encoder kind is an error, not N);
//   - T: Ollama tags vision and none of the row's families runs images;
//   - U: the row is marked unverified;
//   - S: otherwise. Audio is a note, never a status.
func deriveOllama(d *OllamaData, byName map[string]*Family) ([]*OllamaEntry, error) {
	var out []*OllamaEntry
	for _, r := range d.Rows {
		e := &OllamaEntry{OllamaRow: r}
		known, vision := len(r.Families) > 0, false
		for _, f := range r.Families {
			if kind, ok := strings.CutPrefix(f, "encoder:"); ok {
				if !encoderKinds[kind] {
					return nil, fmt.Errorf("ollama row %d (%s): encoder kind %q is not one aikit's encoder loads", r.Rank, r.Tag, kind)
				}
				e.Encoder = true
				continue
			}
			fam := byName[f]
			if fam == nil {
				known = false
				continue
			}
			e.Links = append(e.Links, fam)
			vision = vision || hasVision(fam.Row)
		}
		switch {
		case !known:
			e.Status = OllamaUnsupported
		case slices.Contains(r.Caps, "vision") && !vision:
			e.Status = OllamaTextOnly
		case r.Unverified:
			e.Status = OllamaUnverified
		default:
			e.Status = OllamaSupported
		}
		out = append(out, e)
	}
	return out, nil
}

var tableRow = regexp.MustCompile(`^\|\s*(\d+)\s*\|`)

// snapshotFamilies reads the snapshot's family cell the way ollama.json writes it: "—" is none, " / " separates
// families, "aikit encoder (X)" is encoder:x, and "(decoder as embedder)" is dropped.
func snapshotFamilies(cell string) []string {
	if cell == "—" || cell == "" {
		return []string{}
	}
	var out []string
	for _, f := range strings.Split(cell, " / ") {
		if k, ok := strings.CutPrefix(f, "aikit encoder ("); ok {
			out = append(out, "encoder:"+strings.ToLower(strings.TrimSuffix(k, ")")))
			continue
		}
		out = append(out, strings.TrimSuffix(f, " (decoder as embedder)"))
	}
	return out
}

func splitCells(ln string) []string {
	c := strings.Split(strings.Trim(strings.TrimSpace(ln), "|"), "|")
	for i := range c {
		c[i] = strings.TrimSpace(c[i])
	}
	return c
}

// CheckOllama holds the section to its snapshot (docs/tasks/task-site-2026-09.md S2, S8b). It fails the build when:
//   - the data's rows are not the snapshot table's, row for row (rank, tag, pulls, Ollama's tags, families);
//   - a status derived from today's matrix differs from the snapshot's: the matrix has moved since the snapshot, so
//     its headline is stale and a new dated snapshot is needed;
//   - the four shares recomputed from the data and the matrix differ from the snapshot's Result table, or the headline
//     does not carry the supported and text-only shares, or it is not a Fact cited to that table.
func CheckOllama(root string, in *Inputs) error {
	d := in.Ollama
	if d == nil {
		return fmt.Errorf("ollama: no site/data/ollama.json")
	}
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(d.Snapshot)))
	if err != nil {
		return fmt.Errorf("ollama: the snapshot: %w", err)
	}
	md := string(b)
	var bad []string
	byName := map[string]*Family{}
	for _, r := range in.Rows {
		byName[r.Name] = &Family{Row: r}
	}
	entries, err := deriveOllama(d, byName)
	if err != nil {
		return err
	}

	table, ok := Section(md, "The 60, in page order")
	if !ok {
		return fmt.Errorf("ollama: %s has no \"The 60, in page order\" table", d.Snapshot)
	}
	var snap [][]string
	for _, ln := range strings.Split(table, "\n") {
		if tableRow.MatchString(ln) {
			snap = append(snap, splitCells(ln))
		}
	}
	if len(snap) != len(d.Rows) {
		bad = append(bad, fmt.Sprintf("the snapshot table has %d rows, ollama.json %d", len(snap), len(d.Rows)))
	}
	for i := 0; i < min(len(snap), len(d.Rows)); i++ {
		s, r, e := snap[i], d.Rows[i], entries[i]
		if len(s) != 8 {
			bad = append(bad, fmt.Sprintf("snapshot row %d has %d cells, want 8 (#, tag, pulls, tags, family, status, needs, note)", i+1, len(s)))
			continue
		}
		caps := []string{}
		if s[3] != "" {
			for _, c := range strings.Split(s[3], ",") {
				caps = append(caps, strings.TrimSpace(c))
			}
		}
		pulls, _ := strconv.ParseFloat(s[2], 64)
		switch {
		case s[0] != strconv.Itoa(r.Rank) || s[1] != r.Tag:
			bad = append(bad, fmt.Sprintf("row %d: snapshot has #%s %s, ollama.json #%d %s", i+1, s[0], s[1], r.Rank, r.Tag))
		case pulls != r.PullsM:
			bad = append(bad, fmt.Sprintf("row %d (%s): pulls %s in the snapshot, %v in ollama.json", i+1, r.Tag, s[2], r.PullsM))
		case !slices.Equal(caps, r.Caps):
			bad = append(bad, fmt.Sprintf("row %d (%s): Ollama's tags %q in the snapshot, %q in ollama.json", i+1, r.Tag, caps, r.Caps))
		case !slices.Equal(snapshotFamilies(s[4]), r.Families):
			bad = append(bad, fmt.Sprintf("row %d (%s): families %q in the snapshot, %q in ollama.json", i+1, r.Tag, snapshotFamilies(s[4]), r.Families))
		case s[6] != r.Needs:
			bad = append(bad, fmt.Sprintf("row %d (%s): needs %q in the snapshot, %q in ollama.json", i+1, r.Tag, s[6], r.Needs))
		case e.Status == OllamaUnsupported && r.Needs != "" && byName[r.Needs] != nil:
			bad = append(bad, fmt.Sprintf("row %d (%s): the matrix now has %s, which this row needs; the snapshot's figures are "+
				"a reading of an older matrix: a new snapshot needed (docs/measurements/, dated), not an edit to this one", i+1, r.Tag, r.Needs))
		case s[5] != e.Status:
			bad = append(bad, fmt.Sprintf("row %d (%s): the matrix now makes it %s, the snapshot says %s. The snapshot's figures are a "+
				"reading of an older matrix: a new snapshot needed (docs/measurements/, dated), not an edit to this one",
				i+1, r.Tag, e.Status, s[5]))
		}
	}

	// The Result table, recomputed.
	res, ok := Section(md, "Result")
	if !ok {
		return fmt.Errorf("ollama: %s has no Result section", d.Snapshot)
	}
	var total float64
	tags, pulls := map[string]int{}, map[string]float64{}
	for _, e := range entries {
		tags[e.Status]++
		pulls[e.Status] += e.PullsM
		total += e.PullsM
	}
	share := func(st string) string { return strconv.FormatFloat(100*pulls[st]/total, 'f', 1, 64) + "%" }
	for _, st := range []string{OllamaSupported, OllamaTextOnly, OllamaUnsupported, OllamaUnverified} {
		want := fmt.Sprintf("| %d | %.1f | %s |", tags[st], pulls[st], share(st))
		found := false
		for _, ln := range strings.Split(res, "\n") {
			if strings.HasPrefix(ln, "| "+st+",") && strings.HasSuffix(strings.TrimSpace(ln), want) {
				found = true
			}
		}
		if !found {
			bad = append(bad, fmt.Sprintf("Result table: the %s row should end %q (recomputed from the data and the matrix)", st, want))
		}
	}
	for _, st := range []string{OllamaSupported, OllamaTextOnly} {
		if !strings.Contains(d.Headline, share(st)) {
			bad = append(bad, fmt.Sprintf("the headline does not carry the %s share, %s", st, share(st)))
		}
	}
	cited := false
	for _, f := range in.Claims.Facts {
		if f.Text == d.Headline && f.Source.Path == d.Snapshot && f.Source.Heading == "Result" {
			cited = true
		}
	}
	if !cited {
		bad = append(bad, fmt.Sprintf("the headline is not a Fact in claims.json cited to %s, heading \"Result\"", d.Snapshot))
	}
	if len(bad) > 0 {
		return fmt.Errorf("the Ollama section check failed (%d):\n  %s", len(bad), strings.Join(bad, "\n  "))
	}
	return nil
}

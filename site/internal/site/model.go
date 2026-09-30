package site

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Proof is a family's parity string read as the page reads it: how strong the check is, and its two numbers.
type Proof struct {
	Exp      bool     // registered as experimental
	Core     string   // the string without the "experimental: " prefix
	Tier     string   // released | fixture | shared | other
	Argmax   *float64 // % of positions where the next token matches the reference
	Cos      *float64 // worst-position cosine of the logits
	Via      string   // for a shared path, the family it rides on
	Coherent bool
}

// TierOrder ranks the tiers, strongest first.
var TierOrder = map[string]int{"released": 0, "fixture": 1, "shared": 2, "other": 3}

// TierText is what each tier says on the page.
var TierText = map[string]string{
	"released": "Against the released model",
	"fixture":  "Against a small test model",
	"shared":   "Shares another family's check",
	"other":    "Not recorded",
}

var metricRe = regexp.MustCompile(`([\d.]+)%/([\d.]+)`)

// ParseProof reads a parity string such as "experimental: real-oracle 100.0%/0.98988" or "shared-path: deepseek_v3".
func ParseProof(p string) Proof {
	pr := Proof{Core: p}
	if strings.HasPrefix(p, "experimental: ") {
		pr.Exp, pr.Core = true, strings.TrimPrefix(p, "experimental: ")
	}
	pr.Tier = "other"
	switch {
	case strings.HasPrefix(pr.Core, "full-oracle"), strings.HasPrefix(pr.Core, "real-oracle"):
		pr.Tier = "released"
	case strings.HasPrefix(pr.Core, "tiny-oracle"):
		pr.Tier = "fixture"
	case strings.HasPrefix(pr.Core, "shared-path"):
		pr.Tier = "shared"
		if i := strings.Index(pr.Core, ":"); i >= 0 {
			pr.Via = strings.TrimSpace(pr.Core[i+1:])
		}
	}
	if m := metricRe.FindStringSubmatch(pr.Core); m != nil {
		a, _ := strconv.ParseFloat(m[1], 64)
		c, _ := strconv.ParseFloat(m[2], 64)
		pr.Argmax, pr.Cos = &a, &c
	}
	pr.Coherent = strings.Contains(pr.Core, "+coherent")
	return pr
}

// Fit is a verdict about whether a checkpoint fits a machine, in memory: it says nothing about speed.
type Fit struct {
	Verdict string // fits | tight | no
	Note    string
}

// FitFor is the coarse fits / tight / won't-fit verdict (docs/tasks/task-site-2026-09.md S2). memGB is host RAM and
// gpu is apple, nv8, nv16 or none. bytes is the checkpoint's size; moe and gpuEligible come from the family.
func FitFor(bytes int64, moe, gpuEligible bool, memGB float64, gpu string) Fit {
	s := float64(bytes) / 1e9
	switch gpu {
	case "apple":
		if s <= memGB*0.7 {
			if gpuEligible {
				return Fit{"fits", ""}
			}
			return Fit{"fits", "on the CPU"}
		}
		if moe && s < memGB*1.6 {
			return Fit{"tight", "pages experts from disk"}
		}
		return Fit{"no", ""}
	case "nv8", "nv16":
		vram := 8.0
		if gpu == "nv16" {
			vram = 14
		}
		if gpuEligible && s <= vram*0.85 {
			return Fit{"fits", ""}
		}
		if gpuEligible && moe && s < memGB {
			return Fit{"fits", "experts streamed to the card"}
		}
		if s <= memGB*0.6 {
			return Fit{"fits", "on the CPU"}
		}
		if s < memGB*0.9 {
			return Fit{"tight", "on the CPU"}
		}
		return Fit{"no", ""}
	}
	if s <= memGB*0.6 {
		return Fit{"fits", ""}
	}
	if s <= memGB*0.9 || (moe && s < memGB*1.6) {
		if moe {
			return Fit{"tight", "pages experts from disk"}
		}
		return Fit{"tight", ""}
	}
	return Fit{"no", ""}
}

// Speed is one measured decode rate on one machine (a Claim, read for display).
type Speed struct {
	Tok, Peer *float64
	Verdict   string
	Ratio     string
	Why       string
	Date      string
}

// Checkpoint is one model file a page can offer.
type Checkpoint struct {
	ID, Label, Repo, File, Quant, SHA256 string
	Bytes                                int64
	GoodFor, Needs, Tools, Note          string
	Pinned                               bool
	Pull                                 string
	Fit                                  map[string]Fit // by machine key
	Speed                                map[string]*Speed
	Decision                             *DecisionClaim // measured decision figures; nil = unmeasured
	Family                               *Family
}

// Family is a matrix row plus what the pages derive from it.
type Family struct {
	Row
	P           Proof
	Checkpoints []*Checkpoint
}

// HasGGUF reports whether goinfer can fetch this family (a GGUF loader exists).
func (f *Family) HasGGUF() bool { return strings.Contains(f.Loaders, "GGUF") }

// IsMoE reports whether the family has experts.
func (f *Family) IsMoE() bool { return f.MoE != "dense" }

var sizeTail = regexp.MustCompile(`-(\d+(?:\.\d+)?[bBmM])$`)

// DeriveLabel titles a curated-only checkpoint from its repo, since curated.json carries no label: the repo name
// without its org, "-GGUF" or "-Instruct", with the size set off by a space (Qwen2.5-Coder-1.5B -> "Qwen2.5-Coder 1.5B").
func DeriveLabel(repo string) string {
	n := repo[strings.LastIndex(repo, "/")+1:]
	for _, suf := range []string{"-GGUF", "-gguf"} {
		n = strings.TrimSuffix(n, suf)
	}
	n = strings.NewReplacer("-Instruct", "", "-instruct", "").Replace(n)
	return sizeTail.ReplaceAllString(n, " $1")
}

var quantRe = regexp.MustCompile(`(?i)[-.](q\d[a-z0-9_]*|mxfp4|f16|bf16)\.gguf$`)

func quantOf(file string) string {
	if m := quantRe.FindStringSubmatch(file); m != nil {
		return strings.ToLower(m[1])
	}
	return ""
}

// familyStem is a repo without its size and everything after it, so a curated tier can find the family whose registry
// checkpoint comes from the same model line ("Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF" and the 1.5B share "Qwen/Qwen2.5-Coder").
var stemRe = regexp.MustCompile(`-\d+(?:\.\d+)?[bBmM](?:-.*)?$`)

func familyStem(repo string) string { return stemRe.ReplaceAllString(repo, "") }

// Model is everything the pages render.
type Model struct {
	Families []*Family // matrix order
	Ledger   []*Family // sorted strongest check first
	Vetted   []*Checkpoint
	Machines []Machine
	ByName   map[string]*Family
	Method   string
	Download *Download     // the release binaries; nil when the build had no release data
	Docs     []*DocPage    // the repo documents rendered under /docs/
	Book     []BookChapter // the primer's chapters, for Home
	Counts   Counts
	Ollama   *Ollama // the "Coming from Ollama?" section; nil when the inputs had none
}

// Counts is the strip at the top of the Models page.
type Counts struct{ Families, Released, Fixture, Shared, Vetted int }

// Derive builds the Model from validated inputs.
func Derive(in *Inputs) (*Model, error) {
	m := &Model{Machines: in.Machines, ByName: map[string]*Family{}, Method: in.Claims.Method}
	for _, r := range in.Rows {
		f := &Family{Row: r, P: ParseProof(r.Parity)}
		m.Families = append(m.Families, f)
		m.ByName[f.Name] = f
	}
	mkFit := func(c *Checkpoint, f *Family) {
		c.Fit = map[string]Fit{}
		for _, mc := range m.Machines {
			c.Fit[mc.Key] = FitFor(c.Bytes, f.IsMoE(), f.GPUResident, mc.MemoryGB, mc.GPU)
		}
	}
	// Registry checkpoints: one per family that has one.
	byRepoFile := map[string]bool{}
	for _, f := range m.Families {
		rc := f.Checkpoint
		if rc == nil {
			continue
		}
		c := &Checkpoint{ID: rc.Name, Label: rc.Label, Repo: rc.Repo, File: rc.File, Quant: rc.Quant, SHA256: rc.SHA256,
			Bytes: rc.Bytes, GoodFor: rc.GoodFor, Needs: rc.Needs, Tools: rc.Tools, Pinned: rc.SHA256 != "",
			Pull: "goinfer-chat pull " + rc.Name, Family: f}
		mkFit(c, f)
		f.Checkpoints = append(f.Checkpoints, c)
		byRepoFile[rc.Repo+"/"+rc.File] = true
	}
	// Curated tiers not already a registry checkpoint attach to the family whose checkpoint shares their model line.
	tiers := make([]string, 0, len(in.Curated.Tiers))
	for k := range in.Curated.Tiers {
		tiers = append(tiers, k)
	}
	sort.Strings(tiers)
	for _, k := range tiers {
		t := in.Curated.Tiers[k]
		if byRepoFile[t.Repo+"/"+t.File] {
			continue
		}
		var home *Family
		for _, f := range m.Families {
			if f.Checkpoint != nil && familyStem(f.Checkpoint.Repo) == familyStem(t.Repo) {
				home = f
				break
			}
		}
		if home == nil {
			return nil, fmt.Errorf("curated tier %q (%s) shares a model line with no family's registry checkpoint, so no page can carry it", k, t.Repo)
		}
		q := quantOf(t.File)
		c := &Checkpoint{ID: strings.ToLower(strings.TrimSuffix(DeriveLabel(t.Repo), "")), Label: DeriveLabel(t.Repo), Repo: t.Repo,
			File: t.File, Quant: q, SHA256: t.SHA256, Bytes: t.Bytes, Pinned: t.SHA256 != "", Family: home,
			Note:  "Pinned as a release tier (pull/curated.json), not a registry checkpoint, so it carries no good-for, needs or tools fields.",
			Tools: "not yet measured"}
		c.ID = strings.NewReplacer(" ", "-").Replace(c.ID)
		c.Pull = "goinfer-chat pull " + t.Repo + ":" + q
		mkFit(c, home)
		home.Checkpoints = append(home.Checkpoints, c)
	}
	// Speeds from the claims file, by checkpoint id.
	ck := map[string]*Checkpoint{}
	for _, f := range m.Families {
		sort.SliceStable(f.Checkpoints, func(i, j int) bool { return f.Checkpoints[i].Bytes < f.Checkpoints[j].Bytes })
		for _, c := range f.Checkpoints {
			c.Speed = map[string]*Speed{}
			ck[c.ID] = c
			m.Vetted = append(m.Vetted, c)
		}
	}
	for _, cl := range in.Claims.Claims {
		c, ok := ck[cl.Checkpoint]
		if !ok {
			return nil, fmt.Errorf("claim %q is about checkpoint %q, which no page carries", cl.ID, cl.Checkpoint)
		}
		c.Speed[cl.Machine] = &Speed{Tok: cl.Value, Peer: cl.Peer, Verdict: cl.Verdict, Ratio: cl.Ratio, Why: cl.Why, Date: cl.Date}
	}
	for i := range in.Claims.Decisions {
		d := &in.Claims.Decisions[i]
		c, ok := ck[d.Checkpoint]
		if !ok {
			return nil, fmt.Errorf("decision claim %q is about checkpoint %q, which no page carries", d.ID, d.Checkpoint)
		}
		if c.Decision != nil {
			return nil, fmt.Errorf("checkpoint %q has two decision claims (%s, %s)", c.ID, c.Decision.ID, d.ID)
		}
		c.Decision = d
	}
	sort.SliceStable(m.Vetted, func(i, j int) bool { return m.Vetted[i].Bytes < m.Vetted[j].Bytes })
	// The ledger: strongest check first, then non-experimental, then by the two numbers, then by name.
	m.Ledger = append([]*Family(nil), m.Families...)
	sort.SliceStable(m.Ledger, func(i, j int) bool {
		a, b := m.Ledger[i], m.Ledger[j]
		if TierOrder[a.P.Tier] != TierOrder[b.P.Tier] {
			return TierOrder[a.P.Tier] < TierOrder[b.P.Tier]
		}
		if a.P.Exp != b.P.Exp {
			return !a.P.Exp
		}
		if av, bv := val(a.P.Argmax), val(b.P.Argmax); av != bv {
			return av > bv
		}
		if av, bv := val(a.P.Cos), val(b.P.Cos); av != bv {
			return av > bv
		}
		return strings.ToLower(a.DisplayName) < strings.ToLower(b.DisplayName)
	})
	if in.Ollama != nil {
		es, err := deriveOllama(in.Ollama, m.ByName)
		if err != nil {
			return nil, err
		}
		m.Ollama = &Ollama{OllamaData: in.Ollama, Entries: es}
	}
	m.Counts = Counts{Families: len(m.Families), Vetted: len(m.Vetted)}
	for _, f := range m.Families {
		switch f.P.Tier {
		case "released":
			m.Counts.Released++
		case "fixture":
			m.Counts.Fixture++
		case "shared":
			m.Counts.Shared++
		}
	}
	return m, nil
}

func val(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}

// Gaps is the "What hasn't been shown" list, generated from the data and never written by hand. c may be nil (a family
// with no vetted checkpoint). The machine-specific lines are returned apart, by machine key, because the page shows the
// one for the reader's machine.
func (m *Model) Gaps(f *Family, c *Checkpoint) (common []string, byMachine map[string][]string) {
	add := func(s string) { common = append(common, s) }
	if f.P.Tier == "fixture" {
		add("Only checked against a small test model built from the same wiring, because no released " + f.DisplayName + " was small enough to run on the hardware available. Two families promoted past this stage turned out to have real bugs behind a passing fixture.")
	}
	if f.P.Tier == "shared" {
		name := f.P.Via
		if v := m.ByName[f.P.Via]; v != nil {
			name = v.DisplayName
		}
		add("No check of its own. It runs " + name + "'s code, and " + name + "'s numbers stand in for it.")
	}
	if f.P.Argmax != nil && *f.P.Argmax < 100 {
		add(fmt.Sprintf("Picks the same next token as the reference at %s%% of positions, not all of them.", trimFloat(*f.P.Argmax)))
	}
	if f.P.Exp {
		add("Registered as experimental. It can change, or go, before v1.0.")
	}
	if strings.Contains(f.Modality, "ignored") {
		add("The checkpoint has a vision tower, and goinfer skips it. Text only.")
	}
	if f.Name == "qwen3_vl" {
		add("Images aren't supported yet: only the text decoder is implemented.")
	}
	if !f.GPUResident {
		add("Runs on the CPU only today. Not eligible to live on a GPU.")
	}
	if !f.HasGGUF() {
		add("goinfer can't download it for you: this family only loads from safetensors, which come as several files. That's planned (checkpoint fetch, P1–P9).")
	}
	if len(f.Checkpoints) == 0 {
		add("No vetted checkpoint, so there's no fit verdict, no speed and no tool-calling result for this family.")
	}
	byMachine = map[string][]string{}
	if c != nil {
		if !c.Pinned {
			add("This checkpoint isn't pinned: there's no checksum on file, so the download isn't verified.")
		}
		if strings.Contains(c.Tools, "not yet measured") {
			add("Tool calling hasn't been measured on " + c.Label + ".")
		}
		if strings.Contains(c.Tools, "skip") {
			add("Too small for an agent's full tool list: given twelve tools, it answered instead of calling one.")
		}
		for _, mc := range m.Machines {
			s := c.Speed[mc.Key]
			switch {
			case s == nil || s.Tok == nil:
				byMachine[mc.Key] = append(byMachine[mc.Key], "No graded speed on the "+mc.Short+".")
			case s.Peer == nil:
				byMachine[mc.Key] = append(byMachine[mc.Key], "No same-session Ollama comparison on the "+mc.Short+".")
			}
		}
		if len(c.Speed) == 0 {
			add("No speed measured on any machine.")
		}
	}
	return common, byMachine
}

func trimFloat(f float64) string {
	s := strconv.FormatFloat(f, 'f', 1, 64)
	return strings.TrimSuffix(s, ".0")
}

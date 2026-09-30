package site

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const repoRoot = "../../.." // site/internal/site -> the repository

func needRepo(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(repoRoot, "docs", "capability-matrix.json")); err != nil {
		t.Skip("not inside the goinfer repository")
	}
}

func f64(v float64) *float64 { return &v }

func TestParseProof(t *testing.T) {
	cases := []struct {
		in           string
		tier         string
		exp          bool
		argmax, cos  *float64
		via          string
		coherentWant bool
	}{
		{"full-oracle 100.0%/1.00000", "released", false, f64(100), f64(1), "", false},
		{"real-oracle 77.5%/0.99069", "released", false, f64(77.5), f64(0.99069), "", false},
		{"experimental: real-oracle 100.0%/0.98988", "released", true, f64(100), f64(0.98988), "", false},
		{"experimental: tiny-oracle 100.0%/1.00000 +coherent", "fixture", true, f64(100), f64(1), "", true},
		{"shared-path: deepseek_v3", "shared", false, nil, nil, "deepseek_v3", false},
		{"pending", "other", false, nil, nil, "", false},
	}
	for _, c := range cases {
		p := ParseProof(c.in)
		if p.Tier != c.tier || p.Exp != c.exp || p.Via != c.via || p.Coherent != c.coherentWant {
			t.Errorf("%q: got tier %s exp %v via %q coherent %v", c.in, p.Tier, p.Exp, p.Via, p.Coherent)
		}
		if (p.Argmax == nil) != (c.argmax == nil) || (c.argmax != nil && *p.Argmax != *c.argmax) {
			t.Errorf("%q: argmax %v, want %v", c.in, p.Argmax, c.argmax)
		}
		if (p.Cos == nil) != (c.cos == nil) || (c.cos != nil && *p.Cos != *c.cos) {
			t.Errorf("%q: cos %v, want %v", c.in, p.Cos, c.cos)
		}
	}
}

// The verdicts the mockup showed for its six vetted checkpoints on the machines it knew, from the same inputs, so a
// change to the rule is a visible change to the site.
func TestFitFor(t *testing.T) {
	const gb = int64(1e9)
	cases := []struct {
		name        string
		bytes       int64
		moe, gpu    bool
		mem         float64
		gpuKind     string
		verdict, nt string
	}{
		{"0.5B on the Mac", 491400064, false, true, 16, "apple", "fits", ""},
		{"phi3 (CPU only family) on the Mac", 2393231072, false, false, 16, "apple", "fits", "on the CPU"},
		{"gpt-oss 20B on the Mac", 12109566624, true, true, 16, "apple", "tight", "pages experts from disk"},
		{"gemma 26B on the Mac", 14439363584, true, true, 16, "apple", "tight", "pages experts from disk"},
		{"gemma 26B on the 8 GB card", 14439363584, true, true, 62, "nv8", "fits", "experts streamed to the card"},
		{"0.5B on the 8 GB card", 491400064, false, true, 62, "nv8", "fits", ""},
		{"7B-class on a CPU box of 8 GB", 5 * gb, false, true, 8, "none", "tight", ""},
		{"a 30 GB model on 16 GB apple, dense", 30 * gb, false, true, 16, "apple", "no", ""},
	}
	for _, c := range cases {
		f := FitFor(c.bytes, c.moe, c.gpu, c.mem, c.gpuKind)
		if f.Verdict != c.verdict || f.Note != c.nt {
			t.Errorf("%s: got %+v, want %s / %q", c.name, f, c.verdict, c.nt)
		}
	}
}

func TestDeriveLabel(t *testing.T) {
	for in, want := range map[string]string{
		"Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF": "Qwen2.5-Coder 1.5B",
		"Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF": "Qwen2.5-Coder 0.5B",
		"ggml-org/gpt-oss-20b-GGUF":             "gpt-oss 20b",
	} {
		if got := DeriveLabel(in); got != want {
			t.Errorf("DeriveLabel(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestHasTokenAndRound(t *testing.T) {
	if hasToken("| 1.5B | 128 | 1253.1 / 252.9 |", "253.1") {
		t.Error("253.1 must not be found inside 1253.1")
	}
	if hasToken("252.9 and 253.12", "253.1") {
		t.Error("253.1 must not be found inside 253.12")
	}
	for _, s := range []string{"| 253.1 / 252.9 |", "(253.1)", "goinfer 253.1.", "253.1", "**AHEAD** (1.297)"} {
		tok := "253.1"
		if strings.Contains(s, "1.297") {
			tok = "1.297"
		}
		if !hasToken(s, tok) {
			t.Errorf("%q should contain %s", s, tok)
		}
	}
	for in, want := range map[string]string{"1.815": "1.82", "1.272": "1.27", "0.797": "0.80", "1.059": "1.06", "1.296": "1.30", "0.995": "1.00"} {
		if got, err := round2(in); err != nil || got != want {
			t.Errorf("round2(%s) = %s, %v; want %s", in, got, err, want)
		}
	}
}

func TestSection(t *testing.T) {
	md := "# Top\nintro 1.0\n## A\nunder A 2.0\n### A1\ndeeper 3.0\n## B\nunder B 4.0\n"
	for h, want := range map[string]string{"A": "3.0", "A1": "3.0", "B": "4.0"} {
		s, ok := Section(md, h)
		if !ok || !strings.Contains(s, want) {
			t.Errorf("Section(%q) = %q, %v; want it to hold %s", h, s, ok, want)
		}
	}
	if s, _ := Section(md, "A"); strings.Contains(s, "4.0") {
		t.Error("Section(A) must stop at the next heading of its level")
	}
	if s, _ := Section(md, "A1"); strings.Contains(s, "4.0") || strings.Contains(s, "2.0") {
		t.Error("Section(A1) must hold only its own text")
	}
	if _, ok := Section(md, "nope"); ok {
		t.Error("a missing heading must be reported")
	}
}

func realInputs(t *testing.T) *Inputs {
	t.Helper()
	needRepo(t)
	in, err := LoadInputs(repoRoot)
	if err != nil {
		t.Fatal(err)
	}
	if err := in.Validate(); err != nil {
		t.Fatal(err)
	}
	return in
}

func TestCheckClaims_theRealClaimsHold(t *testing.T) {
	in := realInputs(t)
	if err := CheckClaims(repoRoot, in); err != nil {
		t.Fatal(err)
	}
	if len(in.Claims.Claims) == 0 {
		t.Fatal("no claims: the check would be vacuous")
	}
}

// The claims check has to be able to fail. Each case corrupts one claim and must be refused, naming it. The first is
// the mockup's own error: it showed the 1.5B on CUDA at 253.1 tok/s, the first of three runs, where the record's median
// is 252.9.
func TestCheckClaims_refusesDrift(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(c *Claim)
		want   string
	}{
		{"the mockup's 253.1, the first of three runs and not the median", func(c *Claim) { c.Value = f64(253.1) }, "median"},
		{"runs that are not in the record", func(c *Claim) { c.Runs = "253.1 / 252.8 / 252.9" }, "runs"},
		{"no basis declared", func(c *Claim) { c.Basis = "" }, "basis"},
		{"a peer figure from memory", func(c *Claim) { c.Peer = f64(199.9) }, "peer"},
		{"a wrong raw ratio", func(c *Claim) { c.RatioRaw = "1.400" }, "ratio_raw"},
		{"a display that does not round from the raw ratio", func(c *Claim) { c.Ratio = "1.35×" }, "rounds to"},
		{"a date that is not in the section", func(c *Claim) { c.Date = "2026-01-01" }, "date"},
		{"a heading that is not in the file", func(c *Claim) { c.Source.Heading = "No such heading" }, "heading"},
		{"a file that does not exist", func(c *Claim) { c.Source.Path = "docs/measurements/nope.md" }, "cited file"},
	}
	for _, tc := range cases {
		in := realInputs(t)
		var target *Claim
		for i := range in.Claims.Claims {
			if in.Claims.Claims[i].ID == "qwen2.5-coder-1.5b/cuda/decode" {
				target = &in.Claims.Claims[i]
			}
		}
		if target == nil {
			t.Fatal("the 1.5B CUDA claim is not in claims.json")
		}
		tc.mutate(target)
		err := CheckClaims(repoRoot, in)
		if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), target.ID) {
			t.Errorf("%s: want a failure naming the claim and %q, got %v", tc.name, tc.want, err)
		}
	}
	in := realInputs(t)
	in.Claims.Facts = append(in.Claims.Facts, Fact{Text: "Ollama v9.9.9", Source: in.Claims.Facts[0].Source})
	if err := CheckClaims(repoRoot, in); err == nil || !strings.Contains(err.Error(), "v9.9.9") {
		t.Errorf("a fact the record does not state must fail, got %v", err)
	}
}

func TestValidate_refuses(t *testing.T) {
	base := func() *Inputs {
		return &Inputs{
			Rows:     []Row{{Name: "a", DisplayName: "A", Summary: "one sentence", Tasks: []string{"chat"}, Parity: "full-oracle 100.0%/1.00000"}},
			Machines: []Machine{{Key: "mac", Name: "Mac", MemoryGB: 16, GPU: "apple"}},
		}
	}
	if err := base().Validate(); err != nil {
		t.Fatalf("the base inputs must be valid: %v", err)
	}
	for name, mut := range map[string]func(in *Inputs){
		"no families":              func(in *Inputs) { in.Rows = nil },
		"a duplicate family":       func(in *Inputs) { in.Rows = append(in.Rows, in.Rows[0]) },
		"a family with no summary": func(in *Inputs) { in.Rows[0].Summary = " " },
		"a family with no tasks":   func(in *Inputs) { in.Rows[0].Tasks = nil },
		"an unusable page name":    func(in *Inputs) { in.Rows[0].Name = "Bad Name/x" },
		"a claim on an unlisted machine": func(in *Inputs) {
			in.Claims.Claims = []Claim{{ID: "x", Machine: "toaster", Date: "2026-01-01", Source: Source{"a", "b"}}}
		},
		"a claim with no date": func(in *Inputs) {
			in.Claims.Claims = []Claim{{ID: "x", Machine: "mac", Source: Source{"a", "b"}}}
		},
	} {
		in := base()
		mut(in)
		if err := in.Validate(); err == nil {
			t.Errorf("%s: must be refused", name)
		}
	}
}

func TestDerive_curatedTiers(t *testing.T) {
	rows := []Row{
		{Name: "qwen2", DisplayName: "Qwen2", Summary: "s", Tasks: []string{"chat"}, Parity: "full-oracle 100.0%/1.00000", GPUResident: true,
			Loaders: "safetensors, GGUF", MoE: "dense",
			Checkpoint: &RegCheckpoint{Name: "q05", Label: "Q 0.5B", Repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF", File: "q05.gguf", Quant: "q4_k_m", Bytes: 5e8, SHA256: "aa"}},
		{Name: "other", DisplayName: "Other", Summary: "s", Tasks: []string{"chat"}, Parity: "pending", MoE: "dense"},
	}
	machines := []Machine{{Key: "mac", Name: "Mac", Short: "Mac", MemoryGB: 16, GPU: "apple"}}
	tier := func(repo, file string) Curated {
		var c Curated
		c.Tiers = map[string]struct {
			Repo   string `json:"repo"`
			File   string `json:"file"`
			SHA256 string `json:"sha256"`
			Bytes  int64  `json:"bytes"`
		}{"x": {Repo: repo, File: file, SHA256: "bb", Bytes: 1.1e9}}
		return c
	}
	m, err := Derive(&Inputs{Rows: rows, Machines: machines, Curated: tier("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "q15-q4_k_m.gguf")})
	if err != nil {
		t.Fatal(err)
	}
	q := m.ByName["qwen2"]
	if len(q.Checkpoints) != 2 || q.Checkpoints[0].Label != "Q 0.5B" || q.Checkpoints[1].Label != "Qwen2.5-Coder 1.5B" {
		t.Fatalf("the curated 1.5B tier must attach to the family with the same model line, after the 0.5B: %+v", q.Checkpoints)
	}
	if c := q.Checkpoints[1]; c.Pull != "goinfer-chat pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m" || c.Pinned != true {
		t.Errorf("curated-only pull line / pinned wrong: %q %v", c.Pull, c.Pinned)
	}
	if len(m.Vetted) != 2 || m.Vetted[0].Bytes > m.Vetted[1].Bytes {
		t.Errorf("vetted checkpoints must be smallest first: %+v", m.Vetted)
	}
	// a tier already in the registry is not listed twice
	m, err = Derive(&Inputs{Rows: rows, Machines: machines, Curated: tier("Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF", "q05.gguf")})
	if err != nil || len(m.ByName["qwen2"].Checkpoints) != 1 {
		t.Errorf("a curated tier equal to a registry checkpoint must not duplicate it: %v", err)
	}
	// a tier no family can carry is an error, not a silent omission
	if _, err := Derive(&Inputs{Rows: rows, Machines: machines, Curated: tier("acme/Mystery-9B-GGUF", "m.gguf")}); err == nil {
		t.Error("a curated tier with no family must fail the build")
	}
	// a claim about a checkpoint no page carries is an error
	if _, err := Derive(&Inputs{Rows: rows, Machines: machines, Claims: Claims{Claims: []Claim{{ID: "z", Checkpoint: "ghost", Machine: "mac"}}}}); err == nil {
		t.Error("a claim about a checkpoint with no page must fail the build")
	}
}

func TestLedgerOrder(t *testing.T) {
	rows := []Row{
		{Name: "e", DisplayName: "E", Parity: "experimental: real-oracle 100.0%/0.99"},
		{Name: "s", DisplayName: "S", Parity: "shared-path: a"},
		{Name: "f", DisplayName: "F", Parity: "tiny-oracle 100.0%/1.00000"},
		{Name: "b", DisplayName: "B", Parity: "full-oracle 100.0%/0.99"},
		{Name: "a", DisplayName: "A", Parity: "full-oracle 100.0%/1.00000"},
		{Name: "c", DisplayName: "C", Parity: "full-oracle 99.0%/1.00000"},
	}
	for i := range rows {
		rows[i].Summary, rows[i].Tasks = "s", []string{"chat"}
	}
	m, err := Derive(&Inputs{Rows: rows, Machines: []Machine{{Key: "mac", Name: "Mac", MemoryGB: 16, GPU: "apple"}}})
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, f := range m.Ledger {
		got = append(got, f.Name)
	}
	// released (non-experimental first: a, b by cosine, c by lower agreement), experimental, then fixture, shared
	if want := "a b c e f s"; strings.Join(got, " ") != want {
		t.Errorf("ledger order %v, want %s", got, want)
	}
}

func TestBuild_realRepo(t *testing.T) {
	needRepo(t)
	out := t.TempDir()
	rep, err := Build(repoRoot, out, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	in := realInputs(t)
	if rep.Families != len(in.Rows) {
		t.Fatalf("wrote %d family pages for %d families", rep.Families, len(in.Rows))
	}
	for _, rel := range []string{"index.html", "404.html", "models/index.html", "assets/site.css", "assets/site.js"} {
		if b, err := os.ReadFile(filepath.Join(out, rel)); err != nil || len(b) == 0 {
			t.Errorf("%s missing or empty: %v", rel, err)
		}
	}
	idx, _ := os.ReadFile(filepath.Join(out, "models", "index.html"))
	// the Models page's counts are the data's
	for _, want := range []string{"model families", "vetted checkpoints", "What the checks mean"} {
		if !strings.Contains(string(idx), want) {
			t.Errorf("the Models page lacks %q", want)
		}
	}
	// the pre-overlap 26B figure must not be back on the checkpoint's page
	g, _ := os.ReadFile(filepath.Join(out, "models", "gemma4", "index.html"))
	if strings.Contains(string(g), "16.12") || !strings.Contains(string(g), "40.2") {
		t.Error("the Gemma 4 page must say 40.2 tok/s and not the pre-overlap 16.12")
	}
}

// The output gate has to fail when a page is missing or a figure is off its page.
func TestVerify_failsWhenAFamilyHasNoPage(t *testing.T) {
	needRepo(t)
	out := t.TempDir()
	in := realInputs(t)
	m, err := Derive(in)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Render(out, DefaultConfig(), nil); err != nil {
		t.Fatal(err)
	}
	if err := Verify(out, m, nil, false); err != nil {
		t.Fatalf("a clean build must verify: %v", err)
	}
	victim := m.Families[3].Name
	if err := os.Remove(filepath.Join(out, "models", victim, "index.html")); err != nil {
		t.Fatal(err)
	}
	if err := Verify(out, m, nil, false); err == nil || !strings.Contains(err.Error(), victim) {
		t.Errorf("Verify must name the family with no page (%s), got %v", victim, err)
	}
	// a figure that is on the claim but not on the page
	if err := os.MkdirAll(filepath.Join(out, "models", victim), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "models", victim, "index.html"), []byte(m.Families[3].DisplayName+" What hasn't been shown"), 0o644); err != nil {
		t.Fatal(err)
	}
	m.Families[3].Checkpoints = append(m.Families[3].Checkpoints, &Checkpoint{ID: "ghost", Speed: map[string]*Speed{"mac": {Tok: f64(123.4), Date: "2026-09-25"}}})
	if err := Verify(out, m, nil, false); err == nil || !strings.Contains(err.Error(), "123.4") {
		t.Errorf("Verify must refuse a speed that is not on its page, got %v", err)
	}
}

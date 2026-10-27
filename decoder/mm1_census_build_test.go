package decoder

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

var zeroTime time.Time

func newCensusChild(exe string) *exec.Cmd {
	cmd := exec.Command(exe, "-test.run=^TestDecisionCensus_secondProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(), "GOINFER_CENSUS_CHILD=1")
	return cmd
}

const (
	censusMiB = int64(1) << 20
	censusGiB = int64(1) << 30
	censusTop = int64(1) << 44 // 16 TiB: above every reading a decision can turn on
)

// setCompiled makes name count as linked (on) or not (off) and returns the restore.
func setCompiled(name string, on bool) (restore func()) {
	backendMu.Lock()
	defer backendMu.Unlock()
	prev, had := backendRegistry[name]
	if on {
		backendRegistry[name] = func() (Backend, error) { return nil, fmt.Errorf("census: a linked %s that builds nothing", name) }
	} else {
		delete(backendRegistry, name)
	}
	return func() {
		backendMu.Lock()
		defer backendMu.Unlock()
		if had {
			backendRegistry[name] = prev
		} else {
			delete(backendRegistry, name)
		}
	}
}

// censusBackend is a backend option and whether that backend counts as linked.
type censusBackend struct {
	label, name string
	linked      bool
}

var censusLoaded sync.Map // path+"\x00"+quant -> *Model or error: a family is loaded once per process

func censusLoad(path, quant string) (*Model, error) {
	key := path + "\x00" + quant
	if v, ok := censusLoaded.Load(key); ok {
		if m, ok := v.(*Model); ok {
			return m, nil
		}
		return nil, v.(error)
	}
	m, err := Load(path, Options{Backend: "cpu", Quant: quant})
	if err != nil {
		censusLoaded.Store(key, err)
		return nil, err
	}
	censusLoaded.Store(key, m)
	return m, nil
}

// censusVariant is a family at one window: its own, or the long copy.
type censusVariant struct {
	name, path string
	dir        bool
	cfg        *Config
}

func censusConfig(path string, dir bool) (*Config, error) {
	if dir {
		return loadConfig(os.DirFS(path), "config.json")
	}
	g, err := OpenGGUFMmap(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	return ggufConfig(g)
}

// buildCensus computes every group. The probe fake is installed for the whole run, so no reading of this machine enters it.
func buildCensus(t *testing.T) *census {
	t.Helper()
	fake := &probe.Fake{}
	useProbe(t, fake)
	for _, b := range []string{"cuda", "metal", "webgpu"} {
		defer setCompiled(b, false)()
	}
	c := &census{groups: map[string]string{}, outcomes: map[string]map[string]bool{}}
	zoo := censusMachines(t)

	var variants []censusVariant
	for _, f := range censusFamilies(t) {
		cfg, err := censusConfig(f.path, f.dir)
		if err != nil {
			continue // not a decoder checkpoint the config reader knows (a tower or an encoder): no fit decision reads it
		}
		variants = append(variants, censusVariant{name: f.name, path: f.path, dir: f.dir, cfg: cfg})
		if f.dir {
			long := longVariant(t, f.path)
			lcfg, err := censusConfig(long, true)
			if err != nil {
				t.Fatalf("%s long: %v", f.name, err)
			}
			variants = append(variants, censusVariant{name: f.name + "@131072", path: long, dir: true, cfg: lcfg})
		}
	}
	if len(variants) < 20 {
		t.Fatalf("only %d family variants resolved a config: a census that read little covers little", len(variants))
	}

	phase := func(name string, run func()) {
		start, before := time.Now(), len(c.groups)
		run()
		fmt.Fprintf(os.Stderr, "census: %s: %d groups in %.1fs\n", name, len(c.groups)-before, time.Since(start).Seconds())
	}
	var loaded []censusModel
	phase("D1", func() { censusD1(t, c, fake, zoo, variants) })
	phase("D2", func() { censusD2(t, c, fake, zoo, variants) })
	phase("loads", func() { loaded = censusLoadAll(t, c, fake, variants) })
	phase("D3", func() { censusD3(c, zoo, loaded) })
	phase("D4", func() { censusD4(c, zoo, loaded) })
	phase("D5", func() { censusD5(c, fake, zoo, loaded) })
	phase("D6", func() { censusD6(c, fake, zoo) })
	phase("D7", func() { censusD7(c, variants) })
	return c
}

func zooAvail(zoo []censusMachine) []int64 {
	var out []int64
	for _, m := range zoo {
		out = append(out, m.AvailableBytes)
	}
	return out
}

// parallelEach runs fn(0..n-1) on every core and waits. fn may read the package's seams but must not change them.
func parallelEach(n int, fn func(i int)) {
	var wg sync.WaitGroup
	next := make(chan int)
	for range max(1, runtime.NumCPU()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	for i := range n {
		next <- i
	}
	close(next)
	wg.Wait()
}

func censusD1(t *testing.T, c *census, fake *probe.Fake, zoo []censusMachine, variants []censusVariant) {
	t.Helper()
	censusSetAvail(fake, 0, censusTop) // what fitCheckFor reads; each reading is then set on the check itself
	backends := []censusBackend{{"cpu", "cpu", false}, {"cuda", "cuda", true}, {"cuda-unlinked", "cuda", false}, {"metal", "metal", true}, {"metal-unlinked", "metal", false}}
	for _, be := range backends {
		restore := setCompiled(be.name, be.linked)
		parallelEach(len(variants), func(i int) {
			v := variants[i]
			quants, kvs, streams, moes := []string{"f32", "int4"}, []string{""}, []bool{false}, []bool{false}
			// Stream-weights moves only a .gguf's pricing (the source file, the C' build), and the CPU's KV precision is KVQuant's.
			gguf := []bool{false}
			if !v.dir {
				gguf = []bool{false, true}
			}
			switch be.label {
			case "cpu":
				quants, kvs, streams = []string{"f32", "int8", "int4", "int4mix"}, []string{"", "i8"}, gguf
			case "cuda":
				kvs, streams, moes = []string{"", "f16", "i8"}, gguf, []bool{false, true}
			case "metal":
				kvs = []string{"", "f16", "i8"}
			}
			for _, q := range quants {
				qm, _ := parseQuant(q)
				for _, ctx := range []int{0, 32768} {
					for _, kv := range kvs {
						for _, st := range streams {
							for _, moe := range moes {
								o := Options{Backend: be.name, Quant: q, ResidentContext: ctx, StreamWeights: st, MoECacheExperts: moe}
								switch kv {
								case "f16":
									o.KVPrecision = "f16"
								case "i8":
									o.KVQuant = "i8"
								}
								key := fmt.Sprintf("D1 %s | %s q=%s ctx=%d kv=%s stream=%v moecache=%v", v.name, be.label, q, ctx, kv, st, moe)
								censusD1Group(c, zoo, key, v.path, q, qm, o)
							}
						}
					}
				}
			}
			if be.label == "cpu" {
				qm, _ := parseQuant("int4")
				censusD1Group(c, zoo, "D1 "+v.name+" | cpu q=int4 no-fit-guard", v.path, "int4", qm,
					Options{Backend: "cpu", Quant: "int4", Knobs: &Knobs{knobNoFitGuard: "1"}})
			}
		})
		restore()
	}
}

func censusD1Group(c *census, zoo []censusMachine, key, path, q string, qm quantMode, o Options) {
	f := fitCheckFor(path, q, qm, o)
	at := func(v int64) fitCheck { g := f; g.availBytes = v; return g }
	edges := changePointsBelow(0, 2*f.need()+1, 8, func(v int64) string { return d1Outcome(at(v)) })
	lines := []string{fmt.Sprintf("priced: weight %d kv %d src %d cudaBuild %d expert %d pack %d deviceKV %v ctx %d pinned %v streamable %v noGuard %v",
		f.weightBytes, f.kvBytes, f.srcFileBytes, f.cudaBuildBytes, f.expertBytes, f.packBytes, f.deviceKV, f.effCtx, f.pinned, f.denseStreamable, f.noGuard)}
	for _, v := range edgeReadings(zooAvail(zoo), edges) {
		g := at(v)
		lines = append(lines, d1Line(g))
		c.reached("D1", d1Outcome(g))
		if _, d := g.decide(); d != nil && d.DenseStreamable {
			c.reached("D1", "refuses streamable")
		}
		if g.cudaBuildBytes > 0 && g.need() == g.cudaBuildBytes {
			c.reached("D1", "cuda build peak")
		}
	}
	c.add(key, lines)
}

func d2Outcome(d giwFit) string {
	switch {
	case !d.known:
		return "proceeds unpriced"
	case d.err != nil && d.pinned:
		return "refuses a pin"
	case d.err != nil:
		return "refuses at the floor"
	case d.pinCtx > 0:
		return "auto-pins"
	default:
		return "fits"
	}
}

// censusRealShapeConfigs are the tracked configs of real models that have no weights here.
func censusRealShapeConfigs(t *testing.T) []censusVariant {
	t.Helper()
	var paths []string
	for _, g := range []string{"testdata/laguna_configs/*.json", "../testdata/nemotron-h-configs/*.json", "../testdata/glm_ocr_real_config.json"} {
		ps, _ := filepath.Glob(g)
		paths = append(paths, ps...)
	}
	sort.Strings(paths)
	var out []censusVariant
	for _, p := range paths {
		cfg, err := loadConfig(os.DirFS(filepath.Dir(p)), filepath.Base(p))
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		out = append(out, censusVariant{name: censusName(p) + " (config only)", cfg: cfg})
	}
	if len(out) == 0 {
		t.Fatal("no real-shape configs")
	}
	return out
}

func censusD2(t *testing.T, c *census, fake *probe.Fake, zoo []censusMachine, variants []censusVariant) {
	t.Helper()
	cfgs := append(append([]censusVariant(nil), variants...), censusRealShapeConfigs(t)...)
	backends := []censusBackend{{"cpu", "cpu", false}, {"cuda", "cuda", true}, {"cuda-unlinked", "cuda", false}, {"metal", "metal", true}}
	for _, v := range cfgs {
		for _, be := range backends {
			restore := setCompiled(be.name, be.linked)
			moes := []bool{false}
			if be.label == "cuda" {
				moes = []bool{false, true}
			}
			kvs := []string{"", "f16", "i8"}
			if be.label == "cpu" {
				kvs = []string{"", "i8"}
			}
			for _, ctx := range []int{0, 32768} {
				for _, kv := range kvs {
					for _, moe := range moes {
						o := Options{Backend: be.name, ResidentContext: ctx, MoECacheExperts: moe}
						switch kv {
						case "f16":
							o.KVPrecision = "f16"
						case "i8":
							o.KVQuant = "i8"
						}
						run := func(avail int64) giwFit {
							censusSetAvail(fake, 0, avail)
							cc := *v.cfg
							return decideGIWFit(&cc, o)
						}
						edges := changePointsBelow(0, 2*(run(censusTop).need+giwMemMargin)+1, 8, func(a int64) string { return d2Outcome(run(a)) })
						var lines []string
						for _, a := range edgeReadings(zooAvail(zoo), edges) {
							d := run(a)
							errText := ""
							if d.err != nil {
								errText = d.err.Error()
							}
							lines = append(lines, fmt.Sprintf("avail %d: %s known %v avail %d budget %d need %d pinnedExperts %d ctx %d pinned %v pinCtx %d | err: %s | note: %s",
								a, d2Outcome(d), d.known, d.avail, d.budget, d.need, d.pinnedExperts, d.ctx, d.pinned, d.pinCtx, errText, d.note))
							c.reached("D2", d2Outcome(d))
						}
						c.add(fmt.Sprintf("D2 %s | %s ctx=%d kv=%s moecache=%v", v.name, be.label, ctx, kv, moe), lines)
					}
				}
			}
			restore()
		}
	}
}

// censusModel is a family variant loaded at one quant.
type censusModel struct {
	name string
	m    *Model
}

func censusLoadAll(t *testing.T, c *census, fake *probe.Fake, variants []censusVariant) []censusModel {
	t.Helper()
	censusSetAvail(fake, censusTop, censusTop)
	var out []censusModel
	var failed []string
	for _, v := range variants {
		for _, q := range []string{"f32", "int4"} {
			m, err := censusLoad(v.path, q)
			if err != nil {
				failed = append(failed, v.name+" at "+q)
				continue
			}
			out = append(out, censusModel{name: v.name + " q=" + q, m: m})
		}
	}
	c.add("loads", append([]string{fmt.Sprintf("%d loaded", len(out))}, failed...))
	if len(out) < 40 {
		t.Fatalf("only %d family loads succeeded (failed: %v): a census that loaded little covers little", len(out), failed)
	}
	return out
}

func planLine(p Plan) string {
	return fmt.Sprintf("%s ctx %d slots %d dense %d expertFull %d expertUsed %d kv %d hostCopy %d extra %d free %d need %d | %s",
		p.Placement, p.Ctx, p.Slots, p.DenseBytes, p.ExpertBytesFull, p.ExpertBytesUsed, p.KVBytes, p.HostCopyBytes, p.ExtraBytes, p.FreeBytes, p.NeedBytes(), p.Reason)
}

func censusD3(c *census, zoo []censusMachine, models []censusModel) {
	parallelEach(len(models), func(i int) {
		cm := models[i]
		for _, be := range []string{"cpu", "cuda", "metal", "webgpu"} {
			var zooFree []int64
			for _, z := range zoo {
				switch {
				case be == "cpu":
					zooFree = append(zooFree, z.AvailableBytes)
				case be == "cuda" && z.CUDAFreeBytes != nil:
					zooFree = append(zooFree, *z.CUDAFreeBytes)
				case be == "metal" && z.MetalBudget != nil:
					zooFree = append(zooFree, *z.MetalBudget)
				}
			}
			for _, ctx := range []int{4096, 32768} {
				for _, pinned := range []bool{false, true} {
					for _, kv := range []string{"f32", "f16", "i8"} {
						for _, extra := range []int64{0, 256 * censusMiB} {
							if extra > 0 && (pinned || kv != "f32") {
								continue // a companion's bytes are one term of the need: one cell of them is enough
							}
							req := PlanRequest{Ctx: ctx, CtxPinned: pinned, KVF16: kv == "f16", KVI8: kv == "i8", ExtraBytes: extra}
							summary := func(free int64) string {
								p := cm.m.Plan(be, free, req)
								return fmt.Sprintf("%s %d %d", p.Placement, p.Ctx, p.Slots)
							}
							edges := changePointsBelow(0, 2*cm.m.Plan(be, censusTop, req).NeedBytes()+1, 8, summary)
							var lines []string
							for _, free := range edgeReadings(zooFree, edges) {
								p := cm.m.Plan(be, free, req)
								lines = append(lines, fmt.Sprintf("free %d: %s", free, planLine(p)))
								c.reached("D3", be+" "+p.Placement.String())
								if p.Placement != PlacementDecline && p.Ctx < req.Ctx {
									c.reached("D3", "shrinks the context")
								}
							}
							c.add(fmt.Sprintf("D3 %s | %s ctx=%d pinned=%v kv=%s extra=%d", cm.name, be, ctx, pinned, kv, extra), lines)
						}
					}
				}
			}
		}
	})
}

func d4Outcome(in MetalFitInput, a MetalFit, candidate int) string {
	switch {
	case !in.FreeKnown || strings.HasPrefix(a.Why, "--fit=off"):
		return "off or unknown"
	case strings.HasPrefix(a.Why, "fit:") && a.Ctx >= candidate:
		return "fits at the candidate"
	case strings.HasPrefix(a.Why, "fit:"):
		return "shrinks"
	default:
		return "floor"
	}
}

func censusD4(c *census, zoo []censusMachine, models []censusModel) {
	var zooFree []int64
	for _, z := range zoo {
		if z.MetalBudget != nil {
			zooFree = append(zooFree, *z.MetalBudget)
		}
	}
	parallelEach(len(models), func(i int) {
		cm := models[i]
		candidate := MetalFitDefaultCtx
		if w := cm.m.Config().MaxPositions; w > 0 && candidate > w {
			candidate = w
		}
		for _, slots := range []int{1, 2, 4} {
			for _, lower := range []int{0, 2048, 8192} {
				for _, paged := range []bool{false, true} {
					for _, known := range []bool{true, false} {
						if (paged || !known) && (slots != 1 || lower != 0 || (paged && !known)) {
							continue // paged experts and an unknown budget each answer before the budget is read: one cell each
						}
						in := MetalFitInput{FreeKnown: known, KVSlots: slots, Lower: lower, PagedMoE: paged}
						run := func(free int64) MetalFit { x := in; x.FreeBytes = free; return cm.m.MetalFitCtx(x) }
						var edges []int64
						if known {
							edges = changePointsBelow(0, 2*(int64(slots)*cm.m.ResidentKVBytes("metal", MetalFitDefaultCtx, false, false)+cm.m.ResidentNeedBytes("metal", 0, MetalFitDefaultCtx, false, false)+MetalFitMarginBytes)+1, 8,
								func(free int64) string { a := run(free); return fmt.Sprintf("%d %s", a.Ctx, a.Why) })
						}
						var lines []string
						for _, free := range edgeReadings(zooFree, edges) {
							x := in
							x.FreeBytes = free
							a := cm.m.MetalFitCtx(x)
							lines = append(lines, fmt.Sprintf("free %d: ctx %d slots %d kv %d | %s", free, a.Ctx, a.KVSlots, a.KVBytes, a.Why))
							c.reached("D4", d4Outcome(x, a, candidate))
						}
						c.add(fmt.Sprintf("D4 %s | slots=%d lower=%d paged=%v known=%v", cm.name, slots, lower, paged, known), lines)
					}
				}
			}
		}
	})
}

func censusD5(c *census, fake *probe.Fake, zoo []censusMachine, models []censusModel) {
	for _, cm := range models {
		for _, prompt := range []int{1, 512, 8192} {
			for _, maxTok := range []int{16, 1024} {
				for _, share := range []int{1, 4} {
					run := func(avail int64) error {
						censusSetAvail(fake, 0, avail)
						return cm.m.AdmitPrefillMemoryShare(prompt, maxTok, false, share)
					}
					edges := changePoints(0, censusTop, 4, func(a int64) string {
						if a == 0 {
							return "unknown" // proceeds like an admission, but is not one: the search must see the refusals above it
						}
						return fmt.Sprint(run(a) == nil)
					})
					var lines []string
					for _, a := range edgeReadings(zooAvail(zoo), edges) {
						err := run(a)
						switch {
						case a == 0:
							c.reached("D5", "unknown proceeds")
						case err == nil:
							c.reached("D5", "admitted")
						default:
							c.reached("D5", "refused")
						}
						lines = append(lines, fmt.Sprintf("avail %d: %v", a, err))
					}
					c.add(fmt.Sprintf("D5 %s | prompt=%d max=%d share=%d", cm.name, prompt, maxTok, share), lines)
				}
			}
		}
		var lines []string
		for _, z := range zoo {
			censusSetAvail(fake, z.PhysicalBytes, z.AvailableBytes)
			ctx, kv, w, budget, known := cm.m.FitBudgetSummary()
			lines = append(lines, fmt.Sprintf("%s: ctx %d kv %d weight %d budget %d known %v", z.Name, ctx, kv, w, budget, known))
		}
		c.add("D5 "+cm.name+" | FitBudgetSummary", lines)
	}
}

func censusD6(c *census, fake *probe.Fake, zoo []censusMachine) {
	for _, req := range []int64{0, censusGiB} {
		var lines []string
		for _, z := range zoo {
			censusSetAvail(fake, z.PhysicalBytes, z.AvailableBytes)
			got := resolveWeightCacheBudget(req)
			lines = append(lines, fmt.Sprintf("%s: %d", z.Name, got))
			switch {
			case req > 0:
				c.reached("D6", "explicit")
			case got > 0:
				c.reached("D6", "auto from available")
			default:
				c.reached("D6", "unknown")
			}
		}
		c.add(fmt.Sprintf("D6 requested=%d", req), lines)
	}
}

func censusD7(c *census, variants []censusVariant) {
	var lines []string
	for _, f16 := range []bool{false, true} {
		for _, i8 := range []bool{false, true} {
			lines = append(lines, fmt.Sprintf("f16 %v i8 %v: %d", f16, i8, WebGPUCtxCeiling(f16, i8)))
		}
	}
	c.add("D7 WebGPUCtxCeiling", lines)
	for _, v := range variants {
		cc := *v.cfg
		arch, _, err := resolveArchitecture(&cc)
		if err != nil {
			c.add("D7 "+v.name, []string{"no architecture: " + err.Error()})
			continue
		}
		for _, be := range []string{"cpu", "cuda", "metal", "webgpu"} {
			var lines []string
			for _, prec := range []string{"", "f32", "f16"} {
				for _, q := range []string{"", "i8"} {
					got, why := ResolveKV(be, arch, Options{Backend: be, KVPrecision: prec, KVQuant: q})
					lines = append(lines, fmt.Sprintf("precision %q quant %q: %s | %s", prec, q, got, why))
				}
			}
			c.reached("D7", be)
			c.add(fmt.Sprintf("D7 %s | %s", v.name, be), lines)
		}
	}
}

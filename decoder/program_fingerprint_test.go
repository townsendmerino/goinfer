package decoder

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

const programFingerprintsPath = "../testdata/program_fingerprints.json"

// fixtureProgramFingerprints fingerprints every fixture config.json under testdata that git does not ignore and that resolves to an
// architecture, keyed by its directory relative to the repository root. A config the registry does not take (an
// encoder, a tower, a sampling-only fixture) is not a decoder program and has no entry.
func fixtureProgramFingerprints(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.WalkDir("../testdata", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || d.Name() != "config.json" {
			return nil
		}
		dir := filepath.Dir(path)
		// A gitignored fixture (a local drop CI never has) is not in the table.
		if exec.Command("git", "check-ignore", "-q", path).Run() == nil {
			return nil
		}
		p, perr := programForConfigDir(dir)
		if perr != nil {
			return nil
		}
		out[strings.TrimPrefix(filepath.ToSlash(dir), "../")] = programFingerprint(p)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("no fixture config resolved: the walk covered nothing")
	}
	return out
}

// TestProgramFingerprint_table is VK4's F1: every committed fixture's program fingerprint equals the committed table for
// this GOARCH (-update rewrites this GOARCH's entry), and two computations agree. The table is per architecture because a
// program's float parameters (RoPE frequencies and the like) are computed with math functions whose last bit differs
// between amd64's assembly and arm64's fused multiply-adds.
func TestProgramFingerprint_table(t *testing.T) {
	got := fixtureProgramFingerprints(t)
	again := fixtureProgramFingerprints(t)
	for k, v := range got {
		if again[k] != v {
			t.Fatalf("%s: two computations differ (%s, %s)", k, v, again[k])
		}
	}
	table := map[string]map[string]string{}
	if raw, err := os.ReadFile(programFingerprintsPath); err == nil {
		if err := json.Unmarshal(raw, &table); err != nil {
			t.Fatal(err)
		}
	} else if !*updateMatrix {
		t.Fatalf("read %s: %v (run with -update to create it)", programFingerprintsPath, err)
	}
	arch := runtime.GOARCH
	if *updateMatrix {
		table[arch] = got
		b, err := json.MarshalIndent(table, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(programFingerprintsPath, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d %s fingerprints to %s", len(got), arch, programFingerprintsPath)
		return
	}
	want, ok := table[arch]
	if !ok {
		t.Fatalf("%s has no %s table: run with -update on %s", programFingerprintsPath, arch, arch)
	}
	var diffs []string
	for k, v := range got {
		if want[k] != v {
			diffs = append(diffs, k+" "+v)
		}
	}
	for k := range want {
		if _, ok := got[k]; !ok {
			diffs = append(diffs, k+" (no longer resolves)")
		}
	}
	sort.Strings(diffs)
	if len(diffs) > 0 {
		t.Errorf("%s program fingerprints changed for %d fixture(s); if the change is meant, run -update and say which programs moved:\n  %s",
			arch, len(diffs), strings.Join(diffs, "\n  "))
	}
	t.Logf("%d fixture programs (%s)", len(got), arch)
}

// TestProgramFingerprint_configMatchesWeights is VK4's F2: for every fixture whose weights are present here, the program
// built from its config alone (configExperts) has the fingerprint of the one Load builds from its weights.
func TestProgramFingerprint_configMatchesWeights(t *testing.T) {
	fps := fixtureProgramFingerprints(t)
	dirs := make([]string, 0, len(fps))
	for d := range fps {
		dirs = append(dirs, d)
	}
	sort.Strings(dirs)
	var checked, absent []string
	for _, d := range dirs {
		m, err := Load(filepath.Join("..", d), Options{})
		if err != nil {
			absent = append(absent, d)
			continue
		}
		got := programFingerprint(m.program())
		m.Close()
		checked = append(checked, d)
		if got != fps[d] {
			t.Errorf("%s: the config's program differs from the loaded weights' program", d)
		}
	}
	t.Logf("checked %d fixtures; %d not loadable here: %v", len(checked), len(absent), absent)
	if len(checked) == 0 {
		t.Skip("no fixture weights present: nothing compared")
	}
}

// TestProgramFingerprint_realCheckpoints logs the program fingerprint of every real-checkpoint directory registered in
// testdata/assets.json whose config.json is on this machine, for VK4's record. It asserts only that each resolves.
func TestProgramFingerprint_realCheckpoints(t *testing.T) {
	raw, err := os.ReadFile("../testdata/assets.json")
	if err != nil {
		t.Fatal(err)
	}
	var reg struct {
		Assets []struct {
			Env  string `json:"env"`
			Kind string `json:"kind"`
		} `json:"assets"`
	}
	if err := json.Unmarshal(raw, &reg); err != nil {
		t.Fatal(err)
	}
	found := 0
	for _, a := range reg.Assets {
		if a.Kind != "dir" {
			continue
		}
		dir, err := lookupAsset(a.Env)
		if err != nil {
			continue
		}
		if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
			continue
		}
		p, err := programForConfigDir(dir)
		if err != nil {
			t.Logf("real %s %s: not a decoder program (%v)", a.Env, filepath.Base(dir), err)
			continue
		}
		found++
		t.Logf("real %s %s %s", a.Env, filepath.Base(dir), programFingerprint(p))
	}
	if found == 0 {
		t.Skip("no registered real checkpoint config on this machine")
	}
}

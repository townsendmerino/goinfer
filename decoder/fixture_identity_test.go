package decoder

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The tiny parity checkpoints are gitignored and regenerated per machine by scripts/pin_*.py, but the
// goldens recorded from them are committed — and the pin scripts do not reproduce the same random weights
// across torch/transformers versions. So a box whose checkpoint was re-pinned elsewhere compares the
// committed golden against DIFFERENT weights, and the parity test reports a meaningless cosine (0 for
// gemma3-vl-tiny) that reads exactly like a forward-pass regression. That is what kept five decoder tests
// red on the Mac from 2026-09-18 to 2026-09-24, pre-registered as "known unrelated" and never investigated
// (docs/measurements/tiny-fixture-golden-mismatch-2026-09-24.md).
//
// requireFixtureIdentity fails fast, with the actual diagnosis, when a fixture listed in
// testdata/fixture_identity.json is not the checkpoint its goldens were recorded from. A fixture not in
// the manifest is not checked.
const fixtureIdentityPath = "../testdata/fixture_identity.json"

type fixtureIdentity struct {
	Model    string `json:"model.safetensors"`
	Recorded string `json:"recorded"`
	Pin      string `json:"pin"`
}

func requireFixtureIdentity(t *testing.T, fixtureDir string) {
	t.Helper()
	msg, err := fixtureIdentityMismatch(fixtureIdentityPath, fixtureDir)
	if err != nil {
		t.Fatal(err)
	}
	if msg != "" {
		t.Fatal(msg)
	}
}

// fixtureIdentityMismatch returns a diagnosis when fixtureDir's checkpoint is not the one manifestPath
// records, "" when it is (or the fixture is not listed), and an error when the manifest or checkpoint
// cannot be read.
func fixtureIdentityMismatch(manifestPath, fixtureDir string) (string, error) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", manifestPath, err)
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(raw, &all); err != nil {
		return "", fmt.Errorf("parse %s: %w", manifestPath, err)
	}
	name := filepath.Base(fixtureDir)
	entry, ok := all[name]
	if !ok {
		return "", nil
	}
	var want fixtureIdentity
	if err := json.Unmarshal(entry, &want); err != nil {
		return "", fmt.Errorf("parse %s[%s]: %w", manifestPath, name, err)
	}
	got, err := fileSHA256(filepath.Join(fixtureDir, "model.safetensors"))
	if err != nil {
		return "", fmt.Errorf("hash %s: %w", fixtureDir, err)
	}
	if got == want.Model {
		return "", nil
	}
	return fmt.Sprintf("%s/model.safetensors is not the checkpoint its committed goldens were recorded from "+
		"(sha256 %s…, goldens recorded from %s…, %s).\nThis is a FIXTURE mismatch, not a forward-pass regression: "+
		"copy that box's checkpoint here, or re-pin with %s and commit the regenerated goldens together with the new "+
		"hash in %s.", fixtureDir, got[:12], want.Model[:12], want.Recorded, want.Pin, filepath.Base(manifestPath)), nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// The check must flag a checkpoint that is not the recorded one (naming the fixture and the fix), pass the
// recorded one, and ignore a fixture the manifest does not list.
func TestFixtureIdentityMismatch(t *testing.T) {
	root := t.TempDir()
	mk := func(name, content string) string {
		d := filepath.Join(root, name)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, "model.safetensors"), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return d
	}
	good := mk("listed-tiny", "the recorded weights")
	sum, _ := fileSHA256(filepath.Join(good, "model.safetensors"))
	manifest := filepath.Join(root, "fixture_identity.json")
	if err := os.WriteFile(manifest, []byte(`{"listed-tiny": {"model.safetensors": "`+sum+`", "recorded": "test", "pin": "scripts/pin_x.py"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if msg, err := fixtureIdentityMismatch(manifest, good); err != nil || msg != "" {
		t.Errorf("the recorded checkpoint was flagged: msg=%q err=%v", msg, err)
	}
	if err := os.WriteFile(filepath.Join(good, "model.safetensors"), []byte("re-pinned elsewhere"), 0o644); err != nil {
		t.Fatal(err)
	}
	msg, err := fixtureIdentityMismatch(manifest, good)
	if err != nil || !strings.Contains(msg, "FIXTURE mismatch") || !strings.Contains(msg, "scripts/pin_x.py") {
		t.Errorf("a wrong checkpoint was not flagged with the diagnosis: msg=%q err=%v", msg, err)
	}
	if msg, err := fixtureIdentityMismatch(manifest, mk("unlisted-tiny", "anything")); err != nil || msg != "" {
		t.Errorf("an unlisted fixture was checked: msg=%q err=%v", msg, err)
	}
	// And the real manifest parses and names both fixtures the 2026-09-24 incident involved.
	raw, err := os.ReadFile(fixtureIdentityPath)
	if err != nil {
		t.Fatal(err)
	}
	var real map[string]json.RawMessage
	if err := json.Unmarshal(raw, &real); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"gemma3-vl-tiny", "bailing_hybrid-tiny"} {
		if _, ok := real[n]; !ok {
			t.Errorf("%s missing from %s", n, fixtureIdentityPath)
		}
	}
}

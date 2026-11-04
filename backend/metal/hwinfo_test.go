//go:build darwin

package metal

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestMetalHardwareInfo_factsAndPrivacy: `check --hardware`'s Metal lines (metalHardwareInfo) name the device, the memory
// and the macOS version, and nothing that identifies a person or a machine: no hostname, user name, home directory or
// serial number (internal/hwreport's privacy test covers the report's own sections, not a backend's registered lines).
func TestMetalHardwareInfo_factsAndPrivacy(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skip("no Metal device: " + err.Error())
	}
	lines := metalHardwareInfo()
	all := strings.Join(lines, "\n")
	t.Logf("\n%s", all)
	for _, want := range []string{"device: ", "unified memory: ", "macOS "} {
		if !strings.Contains(all, want) {
			t.Errorf("no %q line in:\n%s", want, all)
		}
	}
	host, _ := os.Hostname()
	home, _ := os.UserHomeDir()
	private := map[string]string{"hostname": strings.TrimSuffix(host, ".local"), "user": os.Getenv("USER"), "home directory": home}
	if out, err := exec.Command("ioreg", "-rd1", "-c", "IOPlatformExpertDevice").Output(); err == nil {
		for l := range strings.SplitSeq(string(out), "\n") {
			if strings.Contains(l, `"IOPlatformSerialNumber"`) {
				if i := strings.LastIndex(l, `"`); i > 0 {
					if j := strings.LastIndex(l[:i], `"`); j >= 0 {
						private["serial number"] = l[j+1 : i]
					}
				}
			}
		}
	}
	for what, v := range private {
		if v != "" && strings.Contains(strings.ToLower(all), strings.ToLower(v)) {
			t.Errorf("the Metal facts contain the %s", what)
		}
	}
}

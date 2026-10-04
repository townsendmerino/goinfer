// Package hwreport prints the block `check --hardware` shows (docs/tasks/task-hardware-coverage-2026-10.md, H3): what a bug report about hardware needs, in one paste.
//
// Nothing is sent anywhere: the block is written to the terminal for the user to paste or not. It leaves out what identifies a person or a machine (no hostname, no user name, no home
// directory, no serial numbers, no network addresses) and a model is named by its file name alone.
package hwreport

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/fitcmd"
)

// Options says which parts to add.
type Options struct {
	// FitModel, when set, adds goinfer-chat's fit decisions for that model (residency, slots, context).
	FitModel string
}

// Write prints the report. It runs the startup self-tests (the CPU one, and each linked GPU backend's) so the block says what they found on THIS machine.
func Write(w io.Writer, o Options) {
	fmt.Fprintln(w, "goinfer hardware report. Nothing here is sent anywhere: paste it into a bug report if you want to.")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "goinfer: %s\n", buildLine())
	fmt.Fprintf(w, "os: %s\n", osLine())
	fmt.Fprintf(w, "cpu: %s; %d logical CPUs\n", cpuModel(), runtime.NumCPU())
	k := linalg.ActiveKernels()
	fmt.Fprintf(w, "cpu kernels (%s): detected %s; in use %s; forced by build tags %s\n", k.Arch, list(k.Detected), list(k.Active), list(k.Forced))
	fmt.Fprintf(w, "memory: %s\n", memoryLine())

	backends := decoder.RegisteredBackends()
	fmt.Fprintf(w, "backends linked into this binary: %s\n", strings.Join(backends, ", "))
	info := decoder.HardwareInfo()
	for _, name := range backends {
		for _, line := range info[name] {
			fmt.Fprintf(w, "  %s: %s\n", name, line)
		}
		if len(info[name]) == 0 { // a backend that gave no device facts still gets its free memory, when it has a probe
			if free, ok := decoder.FreeBytesFor(name); ok {
				fmt.Fprintf(w, "  %s: free memory %s\n", name, gib(free))
			}
		}
	}

	fmt.Fprintln(w, "self-tests (each runs the kernels this process would use against a reference on a small fixed input):")
	for _, r := range decoder.RunSelfTests() {
		fmt.Fprintf(w, "  %s\n", r.Summary())
	}

	if o.FitModel != "" {
		fmt.Fprintf(w, "\nfit decisions for %s:\n", filepath.Base(o.FitModel))
		fitcmd.Run([]string{o.FitModel})
	}
}

// Run implements `<program> check --hardware [--fit MODEL]` for a binary with no server to drive (goinfer-chat). It returns the exit code.
func Run(args []string, program string) int {
	fs := flag.NewFlagSet("check", flag.ContinueOnError)
	hardware := fs.Bool("hardware", false, "print the hardware report (nothing is sent anywhere)")
	fit := fs.String("fit", "", "with --hardware: add the fit decisions for this model (a path or reference)")
	noSelfTest := fs.Bool("no-selftest", false, "skip the startup self-tests")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*hardware {
		fmt.Fprintf(os.Stderr, "usage: %s check --hardware [--fit MODEL] [--no-selftest]\n", program)
		return 2
	}
	if *noSelfTest {
		decoder.SkipSelfTests()
	}
	Write(os.Stdout, Options{FitModel: *fit})
	return 0
}

func buildLine() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return runtime.Version() + " " + runtime.GOOS + "/" + runtime.GOARCH
	}
	rev, dirty := "", ""
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
			if len(rev) > 8 {
				rev = rev[:8]
			}
		case "vcs.modified":
			if s.Value == "true" {
				dirty = "+dirty"
			}
		}
	}
	v := bi.Main.Version
	if v == "" || v == "(devel)" {
		v = "devel"
	}
	if rev != "" {
		v += " (" + rev + dirty + ")"
	}
	return fmt.Sprintf("%s, %s, %s/%s", v, runtime.Version(), runtime.GOOS, runtime.GOARCH)
}

func osLine() string {
	switch runtime.GOOS {
	case "linux":
		name := osReleaseValue("PRETTY_NAME")
		if name == "" {
			name = "linux"
		}
		return name + ", kernel " + strings.TrimSpace(readFile("/proc/sys/kernel/osrelease"))
	case "darwin":
		return strings.TrimSpace(strings.ReplaceAll(run("sw_vers"), "\n", ", "))
	case "windows":
		return strings.TrimSpace(run("cmd", "/c", "ver"))
	}
	return runtime.GOOS
}

func cpuModel() string {
	switch runtime.GOOS {
	case "linux":
		for _, key := range []string{"model name", "Model name", "Hardware", "cpu model", "Processor"} {
			if v := cpuinfoValue(key); v != "" {
				return v
			}
		}
	case "darwin":
		if v := strings.TrimSpace(run("sysctl", "-n", "machdep.cpu.brand_string")); v != "" {
			return v
		}
	case "windows":
		if v := strings.TrimSpace(os.Getenv("PROCESSOR_IDENTIFIER")); v != "" {
			return v
		}
	}
	return "unknown"
}

func memoryLine() string {
	total, avail := decoder.HostRAMBytes(), decoder.HostRAMAvailableBytes()
	if total <= 0 {
		return "host RAM unknown on this platform"
	}
	s := gib(total) + " host RAM"
	if avail > 0 {
		s += ", " + gib(avail) + " available"
	}
	return s
}

func gib(b int64) string { return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30)) }

func list(s []string) string {
	if len(s) == 0 {
		return "none"
	}
	return "[" + strings.Join(s, " ") + "]"
}

func readFile(p string) string {
	b, _ := os.ReadFile(p)
	return string(b)
}

func osReleaseValue(key string) string {
	sc := bufio.NewScanner(strings.NewReader(readFile("/etc/os-release")))
	for sc.Scan() {
		if v, ok := strings.CutPrefix(sc.Text(), key+"="); ok {
			return strings.Trim(v, `"`)
		}
	}
	return ""
}

func cpuinfoValue(key string) string {
	sc := bufio.NewScanner(strings.NewReader(readFile("/proc/cpuinfo")))
	for sc.Scan() {
		if k, v, ok := strings.Cut(sc.Text(), ":"); ok && strings.TrimSpace(k) == key {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

// run executes a read-only system command with a short deadline and returns its stdout, or "" on any failure: a missing tool must never fail the report.
func run(name string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return ""
	}
	return string(out)
}

package main

import (
	"fmt"
	"strings"
)

// The gpu gate's ptx group as a grader (SB12 step 11, docs/tasks/task-subsystems-2026-10.md): its lines and verdicts are a
// function of what the runner (readPTX) recorded: why cuda/build_ptx.sh could not run, the toolchains found, and each
// committed PTX file's outcome when rebuilt at the NVRTC version it records.

const ptxHeader = "4. PTX reproduces from source, each at the NVRTC it records"

// ptxInput is what the ptx group read. Decline is why the script could not run ("" when it ran); Backup is why the
// committed files could not be staged before the rebuild ("" when they were). Files holds every PTX file with a source of
// its own, in name order.
type ptxInput struct {
	Decline    string    `json:"decline,omitempty"`
	Toolchains []string  `json:"toolchains,omitempty"`
	Backup     string    `json:"backup,omitempty"`
	Files      []ptxFile `json:"files,omitempty"`
}

// ptxFile is one PTX file's outcome. Outcome is "ok" (rebuilt byte-identical), "differs", or "unverified" with Why naming
// the reason ("no recorded version", "needs V12.6.85", "build failed at V12.6.85"). KeepErr is why a differing rebuild
// could not be kept for `gate ptx -keep`.
type ptxFile struct {
	Base    string `json:"base"`
	Want    string `json:"want,omitempty"`
	Outcome string `json:"outcome"`
	Why     string `json:"why,omitempty"`
	KeepErr string `json:"keep_err,omitempty"`
}

// gradePTX is the ptx group's verdicts in the order the section prints them. When the script ran, the first verdict's Pre
// lines are the toolchain line and a DIFFERS line for each file that differs. That verdict passes when every file verified
// is identical and at least one was; it fails when a file differs, and when none was verified, which is not a pass with a
// footnote. The verdict counts as ran either way. A second verdict, a counted SKIP, names every file not verified, so a
// partial verification never reads as a full one.
func gradePTX(in ptxInput) []groupVerdict {
	if in.Decline != "" {
		return []groupVerdict{{State: groupSkip, Line: fmt.Sprintf("PTX reproducibility (cuda/build_ptx.sh %s)", in.Decline)}}
	}
	tcs := "none"
	if len(in.Toolchains) > 0 {
		tcs = strings.Join(in.Toolchains, " ")
	}
	pre := []string{"  toolchains available: " + tcs}
	if in.Backup != "" {
		return []groupVerdict{{State: groupSkip, Pre: pre, Line: fmt.Sprintf("PTX reproducibility (cannot stage a backup: %s)", in.Backup)}}
	}
	okN, diff := 0, 0
	var unverified []string
	for _, f := range in.Files {
		switch f.Outcome {
		case "ok":
			okN++
		case "differs":
			diff++
			pre = append(pre, fmt.Sprintf("      DIFFERS: %s.ptx (rebuilt at its recorded V%s)", f.Base, f.Want))
			if f.KeepErr != "" {
				pre = append(pre, fmt.Sprintf("      (could not keep the rebuilt file: %s)", f.KeepErr))
			}
		default:
			unverified = append(unverified, f.Base+"("+f.Why+")")
		}
	}
	total := len(in.Files)
	v := groupVerdict{State: groupFail, Pre: pre, Ran: true}
	switch {
	case diff == 0 && okN > 0:
		v.State, v.Line = groupPass, fmt.Sprintf("%d/%d PTX regenerate byte-identically at their recorded NVRTC", okN, total)
	case diff > 0:
		v.Line = fmt.Sprintf("%d PTX differ from their committed form — the shipped kernels do not match their .cu", diff)
	default:
		v.Line = fmt.Sprintf("PTX reproducibility: no usable NVRTC for any recorded version — zero of %d artifacts verified", total)
	}
	out := []groupVerdict{v}
	if len(unverified) > 0 {
		out = append(out, groupVerdict{State: groupSkip,
			Line: fmt.Sprintf("%d/%d PTX NOT verified on this box (toolchain absent): %s", len(unverified), total, strings.Join(unverified, " "))})
	}
	return out
}

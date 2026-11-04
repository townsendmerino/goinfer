package fitcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// ExitRefused is fit -profile's exit code for a load the recorded machine would refuse; 0 is a load that proceeds (it fits,
// is auto-pinned to a smaller context, or cannot be priced).
const ExitRefused = 4

// machineProfile is the part of a testdata/machines entry the check reads.
type machineProfile struct {
	Name           string `json:"name"`
	AvailableBytes int64  `json:"available_bytes"`
}

// profileVerdict is fit -profile -json's output.
type profileVerdict struct {
	Profile        string            `json:"profile"`
	AvailableBytes int64             `json:"available_bytes"`
	Verdict        string            `json:"verdict"` // fits, capped, refused, unpriced
	Answer         decoder.FitAnswer `json:"answer"`
}

// verdictOf names a FitAnswer's outcome.
func verdictOf(a decoder.FitAnswer) string {
	switch {
	case !a.Fits:
		return "refused"
	case a.Unpriced != "":
		return "unpriced"
	case a.CtxCapped:
		return "capped"
	default:
		return "fits"
	}
}

// runProfile prices path under opts as Load's memory check would on the machine profilePath records, without loading it, and
// writes the answer to w. It returns the exit code.
func runProfile(w io.Writer, path, profilePath string, opts decoder.Options, asJSON bool) int {
	if strings.HasPrefix(path, "hf:") || strings.HasPrefix(path, "demo:") {
		fmt.Fprintf(os.Stderr, "%s: fit -profile prices a local checkpoint; %s is a reference (pull it first)\n", self(), path)
		return 1
	}
	if _, err := os.Stat(path); err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", self(), err)
		return 1
	}
	b, err := os.ReadFile(profilePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", self(), err)
		return 1
	}
	var p machineProfile
	if err := json.Unmarshal(b, &p); err != nil || p.Name == "" {
		fmt.Fprintf(os.Stderr, "%s: %s is not a machine profile (want a testdata/machines entry): %v\n", self(), profilePath, err)
		return 1
	}
	a, err := decoder.FitQueryAvailable(path, opts, p.AvailableBytes)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", self(), err)
		return 1
	}
	v := profileVerdict{Profile: p.Name, AvailableBytes: p.AvailableBytes, Verdict: verdictOf(a), Answer: a}
	if asJSON {
		out, _ := json.Marshal(v)
		fmt.Fprintln(w, string(out))
	} else {
		gib := func(n int64) float64 { return float64(n) / (1 << 30) }
		switch v.Verdict {
		case "refused":
			fmt.Fprintf(w, "fit: REFUSED on %s (%.1f GB available):\n%s\n", p.Name, gib(p.AvailableBytes), a.Decline)
		case "unpriced":
			fmt.Fprintf(w, "fit: UNPRICED on %s: %s\n", p.Name, a.Unpriced)
		case "capped":
			fmt.Fprintf(w, "fit: CAPPED on %s: context %d (model allows %d), needs %.1f GB of a %.1f GB budget\n", p.Name, a.Ctx, a.MaxCtx, gib(a.NeedBytes), gib(a.BudgetBytes))
		default:
			fmt.Fprintf(w, "fit: FITS on %s: needs %.1f GB of a %.1f GB budget\n", p.Name, gib(a.NeedBytes), gib(a.BudgetBytes))
		}
	}
	if v.Verdict == "refused" {
		return ExitRefused
	}
	return 0
}

package decoder

import (
	"os"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
)

const fusedGateUpDefault = cpuengine.FusedGateUpDefault

// envBoolDefault reads a 0/1 boolean env var, returning def when unset or empty.
func envBoolDefault(key string, def bool) bool {
	switch os.Getenv(key) {
	case "1":
		return true
	case "0":
		return false
	default:
		return def
	}
}

// cpuFusedGateUp enables the fused gate+up projection on CPU decode (GOINFER_CPU_FUSED_GATEUP=0/1).
var cpuFusedGateUp = envBoolDefault("GOINFER_CPU_FUSED_GATEUP", fusedGateUpDefault)

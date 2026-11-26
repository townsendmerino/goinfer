package giwcodec

import (
	"fmt"
	"runtime"
)

// Target names the single consumer a .giw bundle is built for, so the writer can choose the
// one on-disk int4 layout that consumer reads. TargetNone ("") means unknown or multi-consumer
// and keeps every int4 tensor canonical (kind 3).
type Target string

const (
	TargetNone     Target = ""
	TargetCPUArm64 Target = "cpu-arm64"
	TargetCPUAmd64 Target = "cpu-amd64"
	TargetMetal    Target = "metal"
	TargetCUDA     Target = "cuda"
	TargetWebGPU   Target = "webgpu"
)

// CarriesKQuant56 reports whether a q4k .giw written for target holds the file Q5_K and Q6_K
// tensors natively (kinds 11 and 12).
func CarriesKQuant56(target Target) bool {
	return target == TargetCPUAmd64 || target == TargetCUDA
}

// TargetForBackend derives the Target for a caller naming backendName as the target consumer.
func TargetForBackend(backendName string, arm64Row4 bool) Target {
	switch backendName {
	case "cpu":
		switch runtime.GOARCH {
		case "arm64":
			if arm64Row4 {
				return TargetCPUArm64
			}
			return TargetNone
		case "amd64":
			return TargetCPUAmd64
		default:
			return TargetNone
		}
	case "metal":
		return TargetMetal
	case "cuda":
		return TargetCUDA
	case "webgpu":
		return TargetWebGPU
	default:
		return TargetNone
	}
}

// ParseTarget parses a -target flag value.
func ParseTarget(s string, arm64Row4 bool) (Target, error) {
	switch Target(s) {
	case TargetCPUArm64, TargetCPUAmd64, TargetMetal, TargetCUDA, TargetWebGPU:
		return Target(s), nil
	case "", "cpu":
		return TargetForBackend("cpu", arm64Row4), nil
	case "canonical":
		return TargetNone, nil
	default:
		return TargetNone, fmt.Errorf("unknown .giw target %q (want cpu-arm64 | cpu-amd64 | metal | cuda | webgpu | cpu | canonical)", s)
	}
}

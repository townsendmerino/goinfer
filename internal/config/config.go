// Package config is the one inventory of goinfer's load and serve configuration: every exported
// decoder.Options field, every GOINFER_* variable production code reads, every per-model knob, and every
// flag the shared loading set and goinfer-serve register, each claimed by exactly one Setting.
// TestInventory holds the table total against those four surfaces (docs/tasks/task-subsystems-2026-10.md
// § "SB10's scope and step 1's gates"). Chat's sampling and REPL flags are not load configuration and are
// not in it.
package config

import "strings"

// Apps is the set of binaries that register a setting's flag.
type Apps uint8

const (
	Chat Apps = 1 << iota
	Serve
)

// EnvRole is how production reads a setting's environment variable: once at startup or as a diagnostic
// (both listed in testdata/env_reads.txt), or as a per-model knob snapshotted at Load and overridable
// through decoder.Options.Knobs.
type EnvRole uint8

const (
	EnvNone EnvRole = iota
	EnvStartup
	EnvDiagnostic
	EnvKnob
)

// Class says who turns a setting: an operator, a diagnostic run (not contract), or code alone.
type Class uint8

const (
	Operator Class = iota
	Diagnostic
	Internal // set by an app or a library caller, never by a person; Reason says by whom
)

// Setting is one knob and every surface it appears on. A field left empty means the setting has no such
// surface.
type Setting struct {
	Name      string            // the flag name when there is a flag, else the variable name, else the Options field
	Flag      string            // flag name, without dashes
	Apps      Apps              // binaries that register Flag
	Env       string            // GOINFER_* variable
	EnvRole   EnvRole           // how Env is read
	Option    string            // exported decoder.Options field
	Default   string            // Flag's DefValue as the flag package prints it, with Env unset
	DefaultOn map[string]string // runtime.GOOS → Default where the platform changes it
	Class     Class
	Reason    string // why an Internal setting has no person-facing surface
	Doc       string // where the setting is documented
	Help      string // Flag's usage text, with the {per-model} and {serve-only} marks help.go describes
	PerModel  string // serve's per-model override key for Flag (--model name=path,key=…)
	ServeHelp string // the text serve puts at {serve-only}
}

// Usage is s.Help as app shows it: serve names the per-model key and adds ServeHelp at their marks, chat drops both.
func (s Setting) Usage(app Apps) string {
	per, extra := "", ""
	if app == Serve {
		extra = s.ServeHelp
		if s.PerModel != "" {
			per = " (per-model override: --model name=path," + s.PerModel + "=…)"
		}
	}
	return strings.NewReplacer("{per-model}", per, "{serve-only}", extra).Replace(s.Help)
}

// DefaultFor is s.Default on goos.
func (s Setting) DefaultFor(goos string) string {
	if v, ok := s.DefaultOn[goos]; ok {
		return v
	}
	return s.Default
}

// Lookup is the setting named name. It panics on a name the schema does not have: callers pass constants, and
// TestInventory holds the schema to the tree.
func Lookup(name string) Setting {
	for _, s := range Settings {
		if s.Name == name {
			return s
		}
	}
	panic("config: no setting " + name)
}

// Settings is the inventory. Order: the shared loading flags, serve's own flags, the startup and
// diagnostic variables, the per-model knobs, then the Options fields only code sets.
var Settings = []Setting{
	{Name: "backend", Flag: "backend", Apps: Chat | Serve, Option: "Backend", Default: "auto", Doc: "docs/flags.md", Help: backendHelp},
	{Name: "quant", Flag: "quant", Apps: Chat | Serve, Option: "Quant", Default: "int4", Doc: "docs/flags.md", Help: "decoder weight quant — the accuracy/speed/RAM tradeoff{per-model}:\n" + quantHelp, PerModel: "quant"},
	{Name: "lora", Flag: "lora", Apps: Chat | Serve, Option: "LoRA", Default: "", Doc: "docs/flags.md", Help: "optional PEFT LoRA adapter dir, merged into the (safetensors) base at load{per-model}", PerModel: "lora"},
	{Name: "kv", Flag: "kv", Apps: Chat | Serve, Option: "KVPrecision", Default: "auto", Doc: "docs/flags.md", Help: kvHelp + "{per-model}", PerModel: "kv"},
	{Name: "ctx", Flag: "ctx", Apps: Chat | Serve, Option: "ResidentContext", Default: "0", Doc: "docs/flags.md", Help: ctxHelp, PerModel: "ctx", ServeHelp: ctxServeHelp},
	{Name: "stream-weights", Flag: "stream-weights", Apps: Chat | Serve, Option: "StreamWeights", Default: "false", Doc: "docs/flags.md", Help: streamWeightsHelp + "{per-model}", PerModel: "stream"},
	{Name: "weight-cache", Flag: "weight-cache", Apps: Chat | Serve, Option: "WeightCacheBytes", Default: "0", Doc: "docs/flags.md", Help: "resident expert-weight budget in GB for -stream-weights (0 = auto, ~half of available RAM){per-model}", PerModel: "weight-cache"},
	{Name: "moe-cache-experts", Flag: "moe-cache-experts", Apps: Chat | Serve, Env: "GOINFER_MOE_CACHE_EXPERTS", EnvRole: EnvKnob, Option: "MoECacheExperts", Default: "false", Doc: "docs/flags.md", Help: moeCacheExpertsHelp},
	{Name: "moe-cache-slots", Flag: "moe-cache-slots", Apps: Chat | Serve, Env: "GOINFER_MOE_CACHE_SLOTS", EnvRole: EnvKnob, Option: "MoECacheSlots", Default: "0", Doc: "docs/flags.md", Help: moeCacheSlotsHelp},
	{Name: "moe-pager", Flag: "moe-pager", Apps: Chat | Serve, Option: "MoEPager", Default: "mmap", DefaultOn: map[string]string{"darwin": "pool"}, Doc: "docs/flags.md", Help: moePagerHelp},
	{Name: "accept-slow", Flag: "accept-slow", Apps: Chat | Serve, Option: "AcceptSlowMoE", Default: "false", Doc: "docs/flags.md", Help: acceptSlowHelp},
	{Name: "embed-int4", Flag: "embed-int4", Apps: Chat | Serve, Option: "EmbedInt4", Default: "true", Doc: "docs/flags.md", Help: embedInt4Help + "{per-model}", PerModel: "embed-int4"},
	{Name: "direct-load", Flag: "direct-load", Apps: Chat | Serve, Env: "GOINFER_GGUF_DIRECT", EnvRole: EnvStartup, Option: "DirectLoad", Default: "false", Doc: "docs/flags.md", Help: directLoadHelp},
	{Name: "fit", Flag: "fit", Apps: Chat | Serve, Option: "DisableFit", Default: "true", Doc: "docs/flags.md", Help: fitHelp},
	{Name: "exact-prefill", Flag: "exact-prefill", Apps: Chat | Serve, Option: "ExactPrefill", Default: "false", Doc: "docs/flags.md", Help: ExactPrefillHelp},
	{Name: "cpu-exact-prefill", Flag: "cpu-exact-prefill", Apps: Chat | Serve, Default: "false", Doc: "docs/flags.md", Help: CPUExactPrefillHelp},
	{Name: "windowed-kv", Flag: "windowed-kv", Apps: Chat | Serve, Option: "ResidentWindowedKV", Default: "true", Doc: "docs/flags.md", Help: WindowedKVHelp},
	{Name: "model", Flag: "model", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "served-model-name", Flag: "served-model-name", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "adapter", Flag: "adapter", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "vision", Flag: "vision", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "vision-device", Flag: "vision-device", Apps: Serve, Default: "auto", Doc: "docs/flags.md"},
	{Name: "vision-quant", Flag: "vision-quant", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "vision-max-pixels", Flag: "vision-max-pixels", Apps: Serve, Default: "0", Doc: "docs/flags.md"},
	{Name: "no-selftest", Flag: "no-selftest", Apps: Serve, Default: "false", Doc: "docs/flags.md"},
	{Name: "embed-model", Flag: "embed-model", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "embed-image-resize", Flag: "embed-image-resize", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "embed-quant", Flag: "embed-quant", Apps: Serve, Default: "f32", Doc: "docs/flags.md"},
	{Name: "embed-served-model-name", Flag: "embed-served-model-name", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "require-backend", Flag: "require-backend", Apps: Serve, Default: "false", Doc: "docs/flags.md"},
	{Name: "kv-quant", Flag: "kv-quant", Apps: Serve, Option: "KVQuant", Doc: "docs/flags.md"},
	{Name: "max-concurrent", Flag: "max-concurrent", Apps: Serve, Default: "4", Doc: "docs/flags.md"},
	{Name: "max-queue", Flag: "max-queue", Apps: Serve, Default: "8", Doc: "docs/flags.md"},
	{Name: "max-inflight", Flag: "max-inflight", Apps: Serve, Default: "128", Doc: "docs/flags.md"},
	{Name: "cpu-batch", Flag: "cpu-batch", Apps: Serve, Option: "CPUBatchDecode", Doc: "docs/flags.md"},
	{Name: "prefill-chunk", Flag: "prefill-chunk", Apps: Serve, Option: "ResidentPrefillChunk", Default: "512", Doc: "docs/flags.md"},
	{Name: "max-body-bytes", Flag: "max-body-bytes", Apps: Serve, Default: "0", Doc: "docs/flags.md"},
	{Name: "kv-host-bytes", Flag: "kv-host-bytes", Apps: Serve, Option: "KVHostBytes", Default: "auto", Doc: "docs/flags.md"},
	{Name: "kv-share", Flag: "kv-share", Apps: Serve, Option: "KVShareOff", Default: "on", Doc: "docs/flags.md"},
	{Name: "kv-sessions", Flag: "kv-sessions", Apps: Serve, Option: "ResidentKVSlots", Default: "4", Doc: "docs/flags.md"},
	{Name: "session-dir", Flag: "session-dir", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "kv-idle-demote", Flag: "kv-idle-demote", Apps: Serve, Default: "0s", Doc: "docs/flags.md"},
	{Name: "kv-demoted-max", Flag: "kv-demoted-max", Apps: Serve, Default: "64", Doc: "docs/flags.md"},
	{Name: "job-dir", Flag: "job-dir", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "spec", Flag: "spec", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "spec-adaptive", Flag: "spec-adaptive", Apps: Serve, Default: "false", Doc: "docs/flags.md"},
	{Name: "drafter", Flag: "drafter", Apps: Serve, Option: "Drafter", Doc: "docs/flags.md"},
	{Name: "drafter-width", Flag: "drafter-width", Apps: Serve, Default: "0", Doc: "docs/flags.md"},
	{Name: "addr", Flag: "addr", Apps: Serve, Default: "127.0.0.1:8080", Doc: "docs/flags.md"},
	{Name: "api-key", Flag: "api-key", Apps: Serve, Env: "GOINFER_API_KEY", EnvRole: EnvStartup, Doc: "docs/flags.md"},
	{Name: "tls-cert", Flag: "tls-cert", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "tls-key", Flag: "tls-key", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "web", Flag: "web", Apps: Serve, Default: "false", Doc: "docs/flags.md"},
	{Name: "allowed-hosts", Flag: "allowed-hosts", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "allow-admin", Flag: "allow-admin", Apps: Serve, Default: "false", Doc: "docs/flags.md"},
	{Name: "admin-socket", Flag: "admin-socket", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "halt-file", Flag: "halt-file", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "halt-exit-code", Flag: "halt-exit-code", Apps: Serve, Default: "0", Doc: "docs/flags.md"},
	{Name: "unload-drain-wait", Flag: "unload-drain-wait", Apps: Serve, Default: "5s", Doc: "docs/flags.md"},
	{Name: "version", Flag: "version", Apps: Serve, Default: "false", Doc: "docs/flags.md"},
	{Name: "decisions-template", Flag: "decisions-template", Apps: Serve, Default: "chat-v1", Doc: "docs/flags.md"},
	{Name: "decisions-calibration", Flag: "decisions-calibration", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "thinking", Flag: "thinking", Apps: Serve, Default: "template", Doc: "docs/flags.md"},
	{Name: "reasoning-format", Flag: "reasoning-format", Apps: Serve, Default: "deepseek", Doc: "docs/flags.md"},
	{Name: "reasoning-budget", Flag: "reasoning-budget", Apps: Serve, Default: "auto", Doc: "docs/flags.md"},
	{Name: "lenient-tool-calls", Flag: "lenient-tool-calls", Apps: Serve, Default: "false", Doc: "docs/flags.md"},
	{Name: "tool-format", Flag: "tool-format", Apps: Serve, Default: "auto", Doc: "docs/flags.md"},
	{Name: "log-requests", Flag: "log-requests", Apps: Serve, Default: "false", Doc: "docs/flags.md"},
	{Name: "log-token-ids", Flag: "log-token-ids", Apps: Serve, Doc: "docs/flags.md"},
	{Name: "carry-ids", Flag: "carry-ids", Apps: Serve, Default: "true", Doc: "docs/flags.md"},
	{Name: "fast-alias", Flag: "fast-alias", Apps: Serve, Default: "true", Doc: "docs/flags.md"},
	{Name: "ids-key-file", Flag: "ids-key-file", Apps: Serve, Doc: "docs/flags.md"},

	{Name: "GOINFER_ATTNFA_DEBUG", Env: "GOINFER_ATTNFA_DEBUG", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/tasks/red-october.md"},
	{Name: "GOINFER_ATTN_KEYS", Env: "GOINFER_ATTN_KEYS", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CPU_FUSED_GATEUP", Env: "GOINFER_CPU_FUSED_GATEUP", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/tasks/red-october.md"},
	{Name: "GOINFER_CUDA_GRAPHS", Env: "GOINFER_CUDA_GRAPHS", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/cuda-graphs-investigation.md"},
	{Name: "GOINFER_CUDA_GRAPHS_ONLY", Env: "GOINFER_CUDA_GRAPHS_ONLY", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/cuda-graphs-investigation.md"},
	{Name: "GOINFER_CUDA_GRAPHS_SYNC", Env: "GOINFER_CUDA_GRAPHS_SYNC", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/cuda-graphs-investigation.md"},
	{Name: "GOINFER_CUDA_GRAPHS_UNSAFE", Env: "GOINFER_CUDA_GRAPHS_UNSAFE", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/cuda-graphs-investigation.md"},
	{Name: "GOINFER_CUDA_VISION_ATTN", Env: "GOINFER_CUDA_VISION_ATTN", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/measurements/vision-tower-downstream-2026-09-21.md"},
	{Name: "GOINFER_DECODE_TIMING", Env: "GOINFER_DECODE_TIMING", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/benchmarks.md"},
	{Name: "GOINFER_DELTANET_TIMING", Env: "GOINFER_DELTANET_TIMING", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/tasks/parked/task-deltanet-cpu-recurrence.md"},
	{Name: "GOINFER_FAKEQUANT", Env: "GOINFER_FAKEQUANT", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/task-gemma4-moe.md"},
	{Name: "GOINFER_FAKEQUANT_ACT", Env: "GOINFER_FAKEQUANT_ACT", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/task-gemma4-moe.md"},
	{Name: "GOINFER_FAKEQUANT_EXPERTS", Env: "GOINFER_FAKEQUANT_EXPERTS", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/task-gemma4-moe.md"},
	{Name: "GOINFER_G4_CAPTURE", Env: "GOINFER_G4_CAPTURE", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_GATE_BACKEND", Env: "GOINFER_GATE_BACKEND", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_GATE_HEARTBEAT", Env: "GOINFER_GATE_HEARTBEAT", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_GATE_MODELS", Env: "GOINFER_GATE_MODELS", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_GATE_SKIP_HEAVY", Env: "GOINFER_GATE_SKIP_HEAVY", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_GATE_SKIP_WEBGPU", Env: "GOINFER_GATE_SKIP_WEBGPU", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_GPU_CAPTURE", Env: "GOINFER_GPU_CAPTURE", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/task-webgpu-nogqa-decode-bug.md"},
	{Name: "GOINFER_HEAVY_PKGS", Env: "GOINFER_HEAVY_PKGS", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_HEAVY_RUN", Env: "GOINFER_HEAVY_RUN", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_HEAVY_TIMEOUT", Env: "GOINFER_HEAVY_TIMEOUT", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MODELS", Env: "GOINFER_MODELS", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MODEL_TMP", Env: "GOINFER_MODEL_TMP", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_CACHE_PROF", Env: "GOINFER_MOE_CACHE_PROF", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/task-moe-streaming.md"},
	{Name: "GOINFER_MOE_PROF_SPLIT", Env: "GOINFER_MOE_PROF_SPLIT", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/audit-metal-2026-09-12.md"},
	{Name: "GOINFER_NORM_ULP_NOISE", Env: "GOINFER_NORM_ULP_NOISE", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/task-webgpu-nogqa-decode-bug.md"},
	{Name: "GOINFER_NVRTC_DIRS", Env: "GOINFER_NVRTC_DIRS", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_P13_OFF", Env: "GOINFER_P13_OFF", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/queue-performance.md"},
	{Name: "GOINFER_REQUIRE_FIXTURES", Env: "GOINFER_REQUIRE_FIXTURES", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_SPLITKV_VSUM_SPLIT", Env: "GOINFER_SPLITKV_VSUM_SPLIT", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/tasks/task-decode-splitkv-attention.md"},
	{Name: "GOINFER_SSM_NOMUL", Env: "GOINFER_SSM_NOMUL", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/ssm-residency-build.md"},
	{Name: "GOINFER_SSM_Q8CPU", Env: "GOINFER_SSM_Q8CPU", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/ssm-residency-build.md"},
	{Name: "GOINFER_SSM_SKIPFFN", Env: "GOINFER_SSM_SKIPFFN", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/completed/ssm-residency-build.md"},
	{Name: "GOINFER_SWAP_GUARD", Env: "GOINFER_SWAP_GUARD", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_TOOL_UNION", Env: "GOINFER_TOOL_UNION", EnvRole: EnvStartup, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_W4A8_BATCH", Env: "GOINFER_W4A8_BATCH", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/tasks/task-cpu-decode-peer-gap-2026-09.md"},
	{Name: "GOINFER_W4A8_SPLITHALF", Env: "GOINFER_W4A8_SPLITHALF", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/tasks/task-int4-layout-2026-09.md"},
	{Name: "GOINFER_WEBGPU_GEMM", Env: "GOINFER_WEBGPU_GEMM", EnvRole: EnvDiagnostic, Class: Diagnostic, Doc: "docs/env-vars.md"},

	{Name: "GOINFER_ATTN_GROUPED", Env: "GOINFER_ATTN_GROUPED", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_ATTN_ROW_TILE", Env: "GOINFER_ATTN_ROW_TILE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_BATCHED_PREFILL", Env: "GOINFER_BATCHED_PREFILL", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CPU_FAST_ATTENTION", Env: "GOINFER_CPU_FAST_ATTENTION", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CUDA_ATTN_FUSED_TILE", Env: "GOINFER_CUDA_ATTN_FUSED_TILE", EnvRole: EnvKnob, Class: Diagnostic, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CUDA_FAST_PREFILL", Env: "GOINFER_CUDA_FAST_PREFILL", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CUDA_FAST_PREFILL_FLOOR", Env: "GOINFER_CUDA_FAST_PREFILL_FLOOR", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CUDA_FLASH_DECODE", Env: "GOINFER_CUDA_FLASH_DECODE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", Env: "GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CUDA_FLASH_DECODE_VERIFY", Env: "GOINFER_CUDA_FLASH_DECODE_VERIFY", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CUDA_MOE_EXPERT_MAJOR", Env: "GOINFER_CUDA_MOE_EXPERT_MAJOR", EnvRole: EnvKnob, Class: Diagnostic, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_CUDA_NO_FUSE", Env: "GOINFER_CUDA_NO_FUSE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_FUSED_ATTENTION", Env: "GOINFER_FUSED_ATTENTION", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_METAL_ALIAS", Env: "GOINFER_METAL_ALIAS", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_METAL_ATTN_FA", Env: "GOINFER_METAL_ATTN_FA", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_METAL_BATCHED_PREFILL", Env: "GOINFER_METAL_BATCHED_PREFILL", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_METAL_DECODE_LANE", Env: "GOINFER_METAL_DECODE_LANE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_METAL_FAST_PREFILL", Env: "GOINFER_METAL_FAST_PREFILL", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_METAL_FAST_PREFILL_FLOOR", Env: "GOINFER_METAL_FAST_PREFILL_FLOOR", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_METAL_FUSED_ATTENTION", Env: "GOINFER_METAL_FUSED_ATTENTION", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_METAL_MOE_SLOTS", Env: "GOINFER_METAL_MOE_SLOTS", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MLA_NAIVE", Env: "GOINFER_MLA_NAIVE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_DMA_OVERLAP", Env: "GOINFER_MOE_DMA_OVERLAP", EnvRole: EnvKnob, Class: Diagnostic, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_EXPERT_MAJOR", Env: "GOINFER_MOE_EXPERT_MAJOR", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_NOCACHE", Env: "GOINFER_MOE_NOCACHE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_PIN_REGISTER", Env: "GOINFER_MOE_PIN_REGISTER", EnvRole: EnvKnob, Class: Diagnostic, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_PREAD", Env: "GOINFER_MOE_PREAD", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_PREAD_CPU", Env: "GOINFER_MOE_PREAD_CPU", EnvRole: EnvKnob, Class: Diagnostic, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_RESIDENCY", Env: "GOINFER_MOE_RESIDENCY", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_MOE_RESIDENCY_SCOPE", Env: "GOINFER_MOE_RESIDENCY_SCOPE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_FIT_DEFAULT", Env: "GOINFER_NO_FIT_DEFAULT", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_FIT_GUARD", Env: "GOINFER_NO_FIT_GUARD", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_GREEDY_FASTPATH", Env: "GOINFER_NO_GREEDY_FASTPATH", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_KVONLY_PREFILL", Env: "GOINFER_NO_KVONLY_PREFILL", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_LORA_CACHE", Env: "GOINFER_NO_LORA_CACHE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_OPTFWD", Env: "GOINFER_NO_OPTFWD", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_RESIDENCY", Env: "GOINFER_NO_RESIDENCY", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_RESIDENT_MEM_GUARD", Env: "GOINFER_NO_RESIDENT_MEM_GUARD", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_RESIDENT_REUSE", Env: "GOINFER_NO_RESIDENT_REUSE", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_SAMPLE_FASTPATH", Env: "GOINFER_NO_SAMPLE_FASTPATH", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_NO_TOPK_FASTPATH", Env: "GOINFER_NO_TOPK_FASTPATH", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_OPTFWD_MAX_TEMP", Env: "GOINFER_OPTFWD_MAX_TEMP", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_PRECISE_MATH", Env: "GOINFER_PRECISE_MATH", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_PREFILL_ATTN_WORKERS", Env: "GOINFER_PREFILL_ATTN_WORKERS", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_PREFILL_CHUNK", Env: "GOINFER_PREFILL_CHUNK", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_PREFILL_IMAGE_CHUNK", Env: "GOINFER_PREFILL_IMAGE_CHUNK", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_SPEC_ADAPTIVE_NEVER_YIELD", Env: "GOINFER_SPEC_ADAPTIVE_NEVER_YIELD", EnvRole: EnvKnob, Class: Diagnostic, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_SPLITKV_ATTN", Env: "GOINFER_SPLITKV_ATTN", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_SPLITKV_MIN_KEYS", Env: "GOINFER_SPLITKV_MIN_KEYS", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},
	{Name: "GOINFER_SSM_RESIDENT", Env: "GOINFER_SSM_RESIDENT", EnvRole: EnvKnob, Class: Operator, Doc: "docs/env-vars.md"},

	{Name: "BackendAuto", Option: "BackendAuto", Class: Internal, Reason: "loadflags sets it when --backend auto chose the backend"},
	{Name: "Knobs", Option: "Knobs", Class: Internal, Reason: "the per-model knob overrides by variable name; each knob is its own setting"},
	{Name: "ResidentKVSlotsDefault", Option: "ResidentKVSlotsDefault", Class: Internal, Reason: "serve sets it when --kv-sessions was not given"},
	{Name: "ExtraResidentBytes", Option: "ExtraResidentBytes", Class: Internal, Reason: "serve prices a --drafter's weights into it"},
	{Name: "ExtraResidentKVPerPosition", Option: "ExtraResidentKVPerPosition", Class: Internal, Reason: "serve prices a --drafter's KV rate into it"},
	{Name: "ActQuantGroup", Option: "ActQuantGroup", Class: Internal, Reason: "library callers only; a q4k load picks 32 itself"},
	{Name: "LoadAbort", Option: "LoadAbort", Class: Internal, Reason: "set by the swap guard around a load, never by a person"},
}

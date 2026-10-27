package decoder

// AdmitFacts is everything the decoder's resident admission reads. Admit reads nothing else: no device, no Model and no
// environment. A Model supplies the facts with admitFacts; the hardware and capability matrices supply them with archAdmitFacts,
// from the arch and the backend alone.
type AdmitFacts struct {
	Arch        *Architecture
	Backend     string // the backend's name, as Backend.Name reports it
	Residency   bool   // the backend can build a resident runner (it implements ResidencyBackend)
	Quant       string // the load's quant option, as Options.Quant spells it
	QuantLabel  string // the precision the weights loaded at (Model.Quant)
	BackendAuto bool   // the backend was chosen by -backend auto
	ActGroup    int    // Options.ActQuantGroup
	ProjsInt4   bool   // the resident projections load as int4 (Model.residentProjsInt4)
	SSMResident bool   // the GOINFER_SSM_RESIDENT knob is set for this load: Granite's bring-up gate, Nemotron-H's int8 opt-in
}

// Admission is Admit's answer: whether the backend may build its resident path, and the gates the published matrices print.
// Why is the load's answer and reads every fact; Shape and Gates are the arch's and the backend's, and read neither the precision
// policy nor a backend's override of the runner shape. A model-free caller (the matrices) reads Shape and Gates, not Why.
type Admission struct {
	// Why is "" when the backend may build; otherwise the first refusal, in the order the load applies the gates.
	Why string
	// Shape is the arch's runner-shape decline, from the arch alone (no backend's override): the matrices' first gate.
	Shape string
	// Gates is every capability gate of a backend that declares its resident features, with this load's activation safety.
	Gates []residentGate
}

// Admit is the decoder's resident admission as a pure function: the backend can build residents at all, the precision policy
// (native Q4_K, -backend auto on Metal, Nemotron's MoE block and int4 default), the runner shape, the activation-group gate, and
// every capability gate of a backend that declares its features. The load path (Model.residencyGate) and the published matrices
// (ResidentEligible, the hardware matrix) read the same answer, and the admission census (TestAdmissionCensus) holds it.
func Admit(f AdmitFacts) Admission {
	key := f.Backend
	if isWebGPUBackend(key) {
		key = "webgpu"
	}
	ad := Admission{Shape: f.Arch.runnerShapeDeclineSSM(f.SSMResident)}
	if _, declared := residentBackendFeatures[key]; declared {
		ad.Gates = residentGateReasons(f.Arch, key, admitActSafe(f, key))
	}
	ad.Why = admitWhy(f, ad)
	return ad
}

// admitActSafe reports whether the load's activations clear ActivationQuantHazard on backend key: per-32 activation scales over
// int8 resident projections (W8A8) do; int4 does not yet. Metal's q4k lane quantizes no activation, so the hazard does not apply.
func admitActSafe(f AdmitFacts, key string) bool {
	if key == "metal" {
		return f.Quant == "q4k"
	}
	return f.ActGroup == 32 && !f.ProjsInt4
}

func admitWhy(f AdmitFacts, ad Admission) string {
	if !f.Residency {
		// Either the CPU backend (expected), or a GPU backend name that resolved to the CPU fallback because the module wasn't
		// built in (`--backend cuda` without `-tags cuda`): NewBackend returns a cpuBackend + a note there.
		return "backend does not implement residency (not built in, or the CPU backend)"
	}
	if f.Quant == "q4k" && f.Backend != "cuda" && f.Backend != "metal" {
		// CUDA has gemv_q4k_g32 and Metal the q4k lane; WebGPU has no Q4_K kernel yet.
		return "--quant q4k (native Q4_K) has no resident kernel on " + f.Backend + " yet (docs/tasks/task-int4-weight-quality-2026-09.md)"
	}
	if why := autoMetalPrecision(f.BackendAuto, f.Backend, f.QuantLabel); why != "" {
		return why
	}
	if a := f.Arch; a.nemotron != nil && a.MoE != nil && !isWebGPUBackend(f.Backend) {
		// Nemotron 3's MoE FFN block has no GPU resident implementation on cuda/metal. The runner-shape gate below declines it too,
		// but only generically; this names the gap (webgpu implements the block).
		return "Nemotron-H MoE FFN block has no GPU resident implementation on cuda/metal (dense Nemotron-H/Nano-9B-v2 is unaffected; webgpu implements it)"
	}
	return admitResident(f, ad)
}

// admitResident is the admission after the backend and precision checks: the runner shape and its precision policy, then, for a
// backend that declares its resident features, the activation-group gate and every capability gate. A backend that declares no
// feature set (an out-of-tree or test backend) keeps the old contract: past the runner shape, its own BuildResident decides.
func admitResident(f AdmitFacts, ad Admission) string {
	if why := admitRunner(f, ad.Shape); why != "" {
		return why
	}
	key := f.Backend
	if isWebGPUBackend(key) {
		key = "webgpu"
	}
	if _, declared := residentBackendFeatures[key]; !declared {
		return ""
	}
	if why := actGroupResidentDecline(key, f.ActGroup, f.Quant); why != "" {
		return why
	}
	if len(ad.Gates) > 0 {
		return ad.Gates[0].Reason
	}
	return ""
}

// admitRunner is the runner-shape gate with its precision policy, given the arch's shape decline: "" when the uniform-layer resident
// runner can express the arch and the load-time precision policy admits it.
func admitRunner(f AdmitFacts, shape string) string {
	a := f.Arch
	eligible := shape == ""
	if !eligible && a.nemotron != nil && a.MoE != nil && isWebGPUBackend(f.Backend) {
		// webgpu implements the Nemotron MoE block (its residency switch and nemoKMoE dispatch), so it overrides the runner shape's
		// blanket MoE-block decline, which stays for cuda and metal: a narrow backend-specific override, not a general escape hatch
		// (docs/tasks/task-gpu-paths-2026-09.md, G7 part 2).
		eligible = true
	}
	if !eligible {
		return "arch is not eligible for the resident decode runner: " + shape
	}
	// Nemotron-H resident is default-on at int4 only: its int4 disagreements with f32 are all at near-tied positions
	// (docs/completed/nemotron-resident.md). int8 stays opt-in behind GOINFER_SSM_RESIDENT.
	if a.nemotron != nil && !f.SSMResident && !f.ProjsInt4 {
		return "Nemotron-H runs resident with int4 projections only by default (load with Quant int4, or set GOINFER_SSM_RESIDENT to allow int8)"
	}
	return ""
}

// archAdmitFacts is the facts of a model-free question, from the arch and the backend alone: a load with the default options on
// a backend that builds residents, and the GOINFER_SSM_RESIDENT knob as the arch reads it.
func archAdmitFacts(a *Architecture, backend string) AdmitFacts {
	return AdmitFacts{Arch: a, Backend: backend, Residency: true, SSMResident: a.ssmResidentKnob()}
}

// admitFacts gathers the facts Admit reads from a loaded model.
func (m *Model) admitFacts() AdmitFacts {
	f := AdmitFacts{Arch: m.w.arch, Quant: m.quant, QuantLabel: m.Quant(), BackendAuto: m.backendAuto, ActGroup: m.actGroup,
		ProjsInt4: m.residentProjsInt4(), SSMResident: m.knobs.get(knobSSMResident) != ""}
	if m.be != nil {
		f.Backend = m.be.Name()
		_, f.Residency = m.be.(ResidencyBackend)
	}
	return f
}

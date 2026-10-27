package mem

// The KV cache precisions. KVNone is a model that keeps no attention KV, which accepts any request; KVAuto leaves the pick to
// the backend.
const (
	KVF32  = "f32"
	KVF16  = "f16"
	KVI8   = "i8"
	KVNone = "n/a"
	KVAuto = "auto"
)

// KVPolicy is a backend's f16 KV rule, bound to one load: why f16 is declined for it ("" when it is not), whether auto means
// f16, and why auto keeps f32 for a model the decline lets through (nil when it never does).
type KVPolicy struct {
	Decline     func() string
	Auto        func() bool
	AutoDecline func() string
}

// KVQuery is what a KV precision is resolved from: the backend, the requests, the model's facts, and the backend's policy.
type KVQuery struct {
	Backend    string // "" or "cpu": the host cache
	Request    string // the resident cache's request (KVAuto when unset)
	CPURequest string // the host cache's request (KVAuto when unset)
	CPUBatchOn bool   // CPU batched decode, which reads f32 KV only
	Drafter    bool   // a block drafter shares the resident's attention kernels

	HasArch    bool // the architecture resolved; every fact below is false without it
	NoKV       bool // no layer keeps attention K/V
	MLA        bool // a latent KV cache
	MoE        bool
	OwnLoop    bool // the family runs its own forward loop
	CPUF16OK   bool // the family's CPU attention reads an f16 cache
	CPUI8OK    bool // the family's CPU attention reads an int8 cache
	CUDAPolicy KVPolicy
}

// ResolveKV is the precision of the KV cache q.Backend builds: KVF32, KVF16, KVI8, KVNone, or "" when nothing can be built as
// asked. reason says why whenever the answer is not what was requested: the backend's pick under auto, or why an explicit
// request cannot be met.
func ResolveKV(q KVQuery) (delivered, reason string) {
	if q.HasArch && q.NoKV {
		return KVNone, "this model keeps no attention KV"
	}
	if q.Backend == "" || q.Backend == "cpu" {
		return resolveCPUKV(q)
	}
	req := q.Request
	switch q.Backend {
	case "cuda":
		return resolveCUDAKV(q, req)
	case "metal":
		if q.HasArch && q.MLA {
			return KVF32, "an MLA model keeps its latent cache in f32 on Metal"
		}
		switch req {
		case KVF32:
			return KVF32, ""
		case KVI8:
			return KVI8, ""
		case KVAuto:
			return KVF16, "auto is f16 on Metal"
		}
		return KVF16, ""
	}
	// WebGPU and any other resident backend build what is asked; auto is f32.
	switch req {
	case KVF16, KVI8, KVF32:
		return req, ""
	}
	return KVF32, "auto is f32 here"
}

func resolveCUDAKV(q KVQuery, req string) (string, string) {
	if !q.HasArch {
		return KVF32, "no architecture"
	}
	p := q.CUDAPolicy
	if p.Decline == nil {
		p.Decline = func() string { return "" } // no policy registered: nothing declines f16
	}
	if p.Auto == nil {
		p.Auto = func() bool { return false } // and auto keeps f32
	}
	switch req {
	case KVF32:
		return KVF32, ""
	case KVI8:
		return KVF32, "CUDA has no int8 KV cache until phase 2"
	case KVF16:
		if q.Drafter {
			return KVF32, "the drafter needs f32 KV until phase 6"
		}
		if why := p.Decline(); why != "" {
			return KVF32, why
		}
		return KVF16, ""
	}
	switch {
	case q.Drafter:
		return KVF32, "the block drafter (--drafter) keeps f32 K/V beside the target's attention kernels"
	case !p.Auto():
		return KVF32, "auto is f32 on CUDA in this build"
	}
	if why := p.Decline(); why != "" {
		return KVF32, why
	}
	if p.AutoDecline != nil {
		if why := p.AutoDecline(); why != "" {
			return KVF32, why
		}
	}
	return KVF16, "auto is f16 on CUDA for this family"
}

func resolveCPUKV(q KVQuery) (string, string) {
	req := q.CPURequest
	if (req == KVF16 || req == KVI8) && q.CPUBatchOn {
		return "", "CPU batched decode (CPUBatchDecode on) reads f32 KV only until phase 4"
	}
	switch req {
	case KVF16:
		if q.HasArch && q.CPUF16OK {
			return KVF16, ""
		}
		return KVF32, cpuKVDecline(q)
	case KVI8:
		if q.HasArch && q.CPUI8OK {
			return KVI8, ""
		}
		return KVF32, cpuKVDecline(q)
	case KVF32:
		return KVF32, ""
	}
	return KVF32, "auto is f32 on the CPU"
}

// cpuKVDecline is why the host cache holds a model at f32 whatever is asked.
func cpuKVDecline(q KVQuery) string {
	switch {
	case !q.HasArch:
		return "no architecture"
	case q.MoE:
		return "MoE families keep an f32 CPU cache until phase 4"
	case q.OwnLoop:
		return "this family's own forward loop keeps an f32 CPU cache until phase 4"
	}
	return "this family's attention keeps an f32 CPU cache until phase 4"
}

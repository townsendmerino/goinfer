package decoder

// fakeSubsystems is the embeddable default for a test resident: every capability absent, as the Noop* types give a
// backend's subsystem wrapper. A fake embeds it, overrides the contract methods it serves (including each
// *Available gate), and returns itself from ResidentSubsystems with selfSubsystems. A type that embeds another fake
// and changes a contract method defines its own ResidentSubsystems, so the subsystem handed out is the outer type.
type fakeSubsystems struct {
	NoopResidentDecode
	NoopResidentPrefill
	NoopResidentKV
	NoopResidentRuntime
}

// fakeContracts is a value that serves all four subsystem contracts at once.
type fakeContracts interface {
	ResidentDecodeSubsystem
	ResidentPrefillSubsystem
	ResidentKVSubsystem
	ResidentRuntimeSubsystem
}

func selfSubsystems(f fakeContracts) ResidentSubsystems {
	return ResidentSubsystems{Decode: f, Prefill: f, KV: f, Runtime: f}
}

// runtimeSubsystems is the ResidentSubsystems of a test host that serves only the runtime contract (a BlockSpec's
// drafter host).
func runtimeSubsystems(rt ResidentRuntimeSubsystem) ResidentSubsystems {
	out := noopResidentSubsystems
	out.Runtime = rt
	return out
}

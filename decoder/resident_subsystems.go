package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/residency"

type (
	NoopResidentDecode  = residency.NoopDecode
	NoopResidentPrefill = residency.NoopPrefill
	NoopResidentKV      = residency.NoopKV
	NoopResidentRuntime = residency.NoopRuntime
)

// SubsystemsOf returns r's four resident subsystems, filling any nil slot with its no-op default.
func SubsystemsOf(r any) ResidentSubsystems {
	if _, ok := r.(ResidentSubsystemProvider); !ok {
		checkSubsystemsOfArg(r)
	}
	return residency.ResolveSubsystems(r)
}

func (m *Model) subsystems() ResidentSubsystems     { return SubsystemsOf(m.resident) }
func (m *Model) resDecode() ResidentDecodeSubsystem { return m.subsystems().Decode }
func (m *Model) resPrefill() ResidentPrefillSubsystem {
	return m.subsystems().Prefill
}
func (m *Model) resKV() ResidentKVSubsystem           { return m.subsystems().KV }
func (m *Model) resRuntime() ResidentRuntimeSubsystem { return m.subsystems().Runtime }

func drafterHostOf(r ResidentForward) (ResidentDrafterHost, bool) { return residency.DrafterHostOf(r) }

var noopResidentSubsystems = residency.NoopSubsystems

func noopLeave() {}

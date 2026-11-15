//go:build goinfer_testhooks

package decoder

import "fmt"

// checkSubsystemsOfArg panics when SubsystemsOf is handed a value that serves a subsystem contract but is not a
// ResidentSubsystemProvider: a subsystem rather than a resident, which resolves to the no-op subsystems and so to "no capability".
// A plain resident (no contract) and nil are legitimate and pass.
func checkSubsystemsOfArg(r any) {
	var serves []string
	if _, ok := r.(ResidentDecodeSubsystem); ok {
		serves = append(serves, "ResidentDecodeSubsystem")
	}
	if _, ok := r.(ResidentPrefillSubsystem); ok {
		serves = append(serves, "ResidentPrefillSubsystem")
	}
	if _, ok := r.(ResidentKVSubsystem); ok {
		serves = append(serves, "ResidentKVSubsystem")
	}
	if _, ok := r.(ResidentRuntimeSubsystem); ok {
		serves = append(serves, "ResidentRuntimeSubsystem")
	}
	if len(serves) > 0 {
		panic(fmt.Sprintf("decoder.SubsystemsOf: %T serves %v but is not a ResidentSubsystemProvider: it is a subsystem, not a resident, and has no capability here; pass the resident", r, serves))
	}
}

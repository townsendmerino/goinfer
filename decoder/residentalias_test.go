package decoder

import (
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/decoder/resident"
)

// TestResidentAlias_namesAreTheLeafTypes holds decoder's resident names to decoder/resident's types: a backend that names
// decoder.ResidentForward and a package that names resident.Forward mean one type, and so does every contract.
func TestResidentAlias_namesAreTheLeafTypes(t *testing.T) {
	for _, c := range []struct {
		name      string
		dec, leaf reflect.Type
	}{
		{"ResidentForward", reflect.TypeFor[ResidentForward](), reflect.TypeFor[resident.Forward]()},
		{"ResidentDecodeSubsystem", reflect.TypeFor[ResidentDecodeSubsystem](), reflect.TypeFor[resident.DecodeSubsystem]()},
		{"ResidentPrefillSubsystem", reflect.TypeFor[ResidentPrefillSubsystem](), reflect.TypeFor[resident.PrefillSubsystem]()},
		{"ResidentKVSubsystem", reflect.TypeFor[ResidentKVSubsystem](), reflect.TypeFor[resident.KVSubsystem]()},
		{"ResidentRuntimeSubsystem", reflect.TypeFor[ResidentRuntimeSubsystem](), reflect.TypeFor[resident.RuntimeSubsystem]()},
		{"ResidentSubsystemProvider", reflect.TypeFor[ResidentSubsystemProvider](), reflect.TypeFor[resident.SubsystemProvider]()},
		{"ResidentSubsystems", reflect.TypeFor[ResidentSubsystems](), reflect.TypeFor[resident.Subsystems]()},
		{"TopKRow", reflect.TypeFor[TopKRow](), reflect.TypeFor[resident.TopKRow]()},
		{"ResidentCaps", reflect.TypeFor[ResidentCaps](), reflect.TypeFor[resident.Caps]()},
	} {
		if c.dec != c.leaf {
			t.Errorf("decoder.%s is %v, the leaf's is %v", c.name, c.dec, c.leaf)
		}
		if c.dec.PkgPath() != "github.com/townsendmerino/goinfer/decoder/resident" {
			t.Errorf("decoder.%s is declared in %q, want decoder/resident", c.name, c.dec.PkgPath())
		}
	}
}

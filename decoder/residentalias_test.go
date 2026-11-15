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
		{"ResidentForward", reflect.TypeOf((*ResidentForward)(nil)).Elem(), reflect.TypeOf((*resident.Forward)(nil)).Elem()},
		{"ResidentDecodeSubsystem", reflect.TypeOf((*ResidentDecodeSubsystem)(nil)).Elem(), reflect.TypeOf((*resident.DecodeSubsystem)(nil)).Elem()},
		{"ResidentPrefillSubsystem", reflect.TypeOf((*ResidentPrefillSubsystem)(nil)).Elem(), reflect.TypeOf((*resident.PrefillSubsystem)(nil)).Elem()},
		{"ResidentKVSubsystem", reflect.TypeOf((*ResidentKVSubsystem)(nil)).Elem(), reflect.TypeOf((*resident.KVSubsystem)(nil)).Elem()},
		{"ResidentRuntimeSubsystem", reflect.TypeOf((*ResidentRuntimeSubsystem)(nil)).Elem(), reflect.TypeOf((*resident.RuntimeSubsystem)(nil)).Elem()},
		{"ResidentSubsystemProvider", reflect.TypeOf((*ResidentSubsystemProvider)(nil)).Elem(), reflect.TypeOf((*resident.SubsystemProvider)(nil)).Elem()},
		{"ResidentSubsystems", reflect.TypeOf(ResidentSubsystems{}), reflect.TypeOf(resident.Subsystems{})},
		{"TopKRow", reflect.TypeOf(TopKRow{}), reflect.TypeOf(resident.TopKRow{})},
		{"ResidentCaps", reflect.TypeOf(ResidentCaps{}), reflect.TypeOf(resident.Caps{})},
	} {
		if c.dec != c.leaf {
			t.Errorf("decoder.%s is %v, the leaf's is %v", c.name, c.dec, c.leaf)
		}
		if c.dec.PkgPath() != "github.com/townsendmerino/goinfer/decoder/resident" {
			t.Errorf("decoder.%s is declared in %q, want decoder/resident", c.name, c.dec.PkgPath())
		}
	}
}

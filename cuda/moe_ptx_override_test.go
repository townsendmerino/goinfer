//go:build cuda && goinfer_testhooks

package cuda

import "os"

// moePTXOrOverride returns the embedded moe.ptx, or a file named by GOINFER_MOE_PTX_FILE.
//
// A9-SPEC needs the reservation and the launch demand measured at a DIFFERENT MOE_MAX_E, and the
// standing constraint is that frozen artifacts are not regenerated. So the scratch build is compiled
// to a separate file and pointed at from here; cuda/testdata/moe.ptx is never touched.
func moePTXOrOverride() []byte {
	if p := os.Getenv("GOINFER_MOE_PTX_FILE"); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			panic("GOINFER_MOE_PTX_FILE unreadable: " + err.Error())
		}
		return b
	}
	return moePTX
}

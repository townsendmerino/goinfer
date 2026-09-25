package decoder

import (
	"os"
	"runtime"
	"testing"
)

// The CPU expert pager's mode resolves in one place, in this order: Options.MoEPager, then
// GOINFER_MOE_PREAD_CPU, then MoEPagerDefault. A library caller used to get mmap on darwin while
// serve got pool, because serve applied its default by setting the env var.
func TestResolveMoEPagerPool(t *testing.T) {
	t.Setenv("GOINFER_MOE_PREAD_CPU", "")
	os.Unsetenv("GOINFER_MOE_PREAD_CPU")
	if got, want := resolveMoEPagerPool("", os.Getenv("GOINFER_MOE_PREAD_CPU")), MoEPagerDefault(runtime.GOOS) == "pool"; got != want {
		t.Errorf("unset everything: pool=%v, want the platform default %v", got, want)
	}
	if !resolveMoEPagerPool("pool", os.Getenv("GOINFER_MOE_PREAD_CPU")) || resolveMoEPagerPool("mmap", os.Getenv("GOINFER_MOE_PREAD_CPU")) {
		t.Error("an explicit Options.MoEPager must win")
	}
	t.Setenv("GOINFER_MOE_PREAD_CPU", "1")
	if !resolveMoEPagerPool("", os.Getenv("GOINFER_MOE_PREAD_CPU")) {
		t.Error("GOINFER_MOE_PREAD_CPU=1 with no option: want pool")
	}
	if resolveMoEPagerPool("mmap", os.Getenv("GOINFER_MOE_PREAD_CPU")) {
		t.Error("an explicit option must beat the env var")
	}
	t.Setenv("GOINFER_MOE_PREAD_CPU", "0")
	if resolveMoEPagerPool("", os.Getenv("GOINFER_MOE_PREAD_CPU")) {
		t.Error("GOINFER_MOE_PREAD_CPU=0 with no option: want mmap")
	}
	for goos, want := range map[string]string{"darwin": "pool", "linux": "mmap", "windows": "mmap"} {
		if got := MoEPagerDefault(goos); got != want {
			t.Errorf("MoEPagerDefault(%q) = %q, want %q", goos, got, want)
		}
	}
}

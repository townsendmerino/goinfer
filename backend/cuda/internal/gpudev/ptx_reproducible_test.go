package gpudev

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// ptxArch must match build_ptx.sh's default; the artifacts are built for it.
const ptxArch = "compute_75"

// nvrtcVersionRe pulls "12.9.86" out of a PTX header line:
//
//	// Cuda compilation tools, release 12.9, V12.9.86
var nvrtcVersionRe = regexp.MustCompile(`Cuda compilation tools, release [\d.]+, V([\d.]+)`)

func ptxToolchainVersion(b []byte) string {
	if m := nvrtcVersionRe.FindSubmatch(b); m != nil {
		return string(m[1])
	}
	return ""
}

// findNVRTC mirrors build_ptx.sh's discovery so the test compiles through exactly the
// path the build script does. NVRTC_LIB overrides, as it does there.
func findNVRTC() (libDir, incDir string) {
	first := func(cands ...string) string {
		for _, c := range cands {
			if matches, _ := filepath.Glob(c); len(matches) > 0 {
				return matches[0]
			}
		}
		return ""
	}
	home := os.Getenv("HOME")
	if libDir = os.Getenv("NVRTC_LIB"); libDir == "" {
		if so := first(
			"/usr/local/cuda/lib64/libnvrtc.so.12",
			"/usr/lib64/libnvrtc.so.12",
			home+"/cuda-toolkit/targets/x86_64-linux/lib/libnvrtc.so.12",
			home+"/.venv*/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
			"../*venv*/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
			"./*venv*/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
		); so != "" {
			libDir = filepath.Dir(so)
		}
	}
	if incDir = os.Getenv("CUDA_INC"); incDir == "" {
		// gemv_quant.cu includes cuda_fp16.h, so this one is not optional for it.
		if h := first(
			"/usr/local/cuda/include/cuda_fp16.h",
			"/usr/include/cuda_fp16.h",
			home+"/cuda-toolkit/targets/x86_64-linux/include/cuda_fp16.h",
			home+"/.venv*/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda_fp16.h",
			"../*venv*/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda_fp16.h",
			"./*venv*/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda_fp16.h",
		); h != "" {
			incDir = filepath.Dir(h)
		}
	}
	return libDir, incDir
}

// TestPTXReproducible rebuilds every shipped .ptx from its .cu and requires the bytes to
// match exactly.
//
// WHY. The PTX is a build-time artifact that ships in the binary via //go:embed, and an
// artifact you cannot regenerate is one you cannot review or audit. goinfer's glue.ptx
// carried a symbol its glue.cu no longer generated for MONTHS, undetected, because
// nothing tied the artifact to its source. aikit ships four such artifacts under four
// CUDA consumers; before this test, nothing tied any of them to anything.
//
// It is also the assertion that makes the histogram below meaningful: the histogram reads
// the COMMITTED PTX, and this test is what establishes that the committed PTX is what the
// committed source actually compiles to.
//
// THE HARNESS WAS VALIDATED, AND HERE IS HOW — because a green reproducibility test is a
// claim like any other, and a comparison that cannot fail would look identical to a clean
// tree. Mutation-checked on 2026-07-31 (nvidia-rtx2070s, NVRTC 12.9.86) three ways:
//
//   - Changing __fmaf_rn -> __fmul_rn in a COMMENT produced byte-identical PTX. Correct:
//     comments do not reach codegen. This is also what lets the BIT-IDENTITY-CONTRACT
//     declarations be added to the .cu headers without invalidating a single artifact.
//   - Changing the same token in real code (gemv_quant.cu's `facc = __fmaf_rn(...)` ->
//     a bare `facc + p0*s`) produced DIFFER, at the expected file.
//   - Restoring the source produced AGREE again.
//
// A byte difference here means one of two things, and the message says which to check
// first: the .ptx was edited by hand or left stale after a .cu change (the glue.ptx
// failure), or the toolchain moved. The version guard below rules out the second before
// the comparison runs, so a reported DIFFER is drift.
//
// On a diff, split bodies on the `}` terminator rather than entry-to-entry — an
// entry-to-entry splitter mis-attributes trailing metadata and produced a residual false
// alarm in goinfer.
func TestPTXReproducible(t *testing.T) {
	libDir, incDir := findNVRTC()
	if libDir == "" {
		msg := "NVRTC not found (install: pip install nvidia-cuda-nvrtc-cu12==12.9.86, or set NVRTC_LIB)"
		// A skipping test is a passing test. CI sets this so the skip cannot hide.
		if os.Getenv("AIKIT_REQUIRE_PTX_REPRO") != "" {
			t.Fatal(msg + " — AIKIT_REQUIRE_PTX_REPRO is set, so this skip is a failure")
		}
		t.Skip(msg)
	}
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found; nvrtc_compile.py needs it")
	}

	for _, dir := range kernelDirs {
		srcs, err := filepath.Glob(filepath.Join(dir, "*.cu"))
		if err != nil {
			t.Fatalf("glob %s: %v", dir, err)
		}
		for _, src := range srcs {
			name := strings.TrimSuffix(filepath.Base(src), ".cu")
			committedPath := filepath.Join(dir, "testdata", name+".ptx")
			committed, err := os.ReadFile(committedPath)
			if err != nil {
				t.Errorf("%s has no committed PTX at %s: %v", src, committedPath, err)
				continue
			}
			t.Run(name, func(t *testing.T) {
				out := filepath.Join(t.TempDir(), name+".ptx")
				cmd := exec.Command("python3", "nvrtc_compile.py", src, out, ptxArch)
				if incDir != "" {
					cmd.Args = append(cmd.Args, incDir)
				}
				cmd.Env = append(os.Environ(),
					"NVRTC_SO="+filepath.Join(libDir, "libnvrtc.so.12"),
					"LD_LIBRARY_PATH="+libDir+string(os.PathListSeparator)+os.Getenv("LD_LIBRARY_PATH"))
				if b, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("nvrtc_compile.py %s: %v\n%s", src, err, b)
				}
				rebuilt, err := os.ReadFile(out)
				if err != nil {
					t.Fatalf("read rebuilt PTX: %v", err)
				}

				// Version guard FIRST: comparing bytes across toolchain versions would
				// report drift that is really a version difference. Rule that out, so a
				// DIFFER below can only mean the artifact and its source disagree.
				wantVer, gotVer := ptxToolchainVersion(committed), ptxToolchainVersion(rebuilt)
				if wantVer == "" {
					t.Fatalf("%s records no NVRTC version — cannot verify reproducibility", committedPath)
				}
				if wantVer != gotVer {
					t.Skipf("committed PTX was built by NVRTC %s but this toolchain is %s — "+
						"reproducibility is only meaningful at the recorded version; "+
						"install %s or set NVRTC_LIB", wantVer, gotVer, wantVer)
				}

				if string(committed) == string(rebuilt) {
					t.Logf("AGREE — %s reproduces %s byte-for-byte at NVRTC %s (%d bytes)",
						src, committedPath, wantVer, len(committed))
					return
				}
				t.Errorf("DIFFER — %s does not reproduce %s at NVRTC %s (committed %d bytes, rebuilt %d). "+
					"Either the .ptx is stale or hand-edited (regenerate with ./build_ptx.sh %s), or the "+
					"source changed without regenerating. Split bodies on the '}' terminator to find which "+
					"kernels moved — entry-to-entry mis-attributes trailing metadata.",
					src, committedPath, wantVer, len(committed), len(rebuilt), name)
			})
		}
	}
}

// fmaHistogram counts the float instructions whose mix the bit-identity contract fixes.
func fmaHistogram(ptx []byte) (fma, mul int) {
	for line := range strings.SplitSeq(string(ptx), "\n") {
		switch {
		case strings.Contains(line, "fma.rn.f32"):
			fma++
		case strings.Contains(line, "mul.rn.f32"), strings.Contains(line, "mul.f32"):
			mul++
		}
	}
	return fma, mul
}

// TestGemvQuantFMAHistogram verifies the compiler HONOURED what the source lint demands.
//
// The lint removes the compiler's discretion at the source; this checks what actually came
// out the other end. It is the assertion that proved the original defect: gemv_w4a8_fwd /
// gemv_w8a8_fwd compiled to 11 fma + 3 mul (mul+add — the compiler declining to fuse)
// while goinfer's batched counterpart compiled to 192 fma + 0 mul, and that mismatch was
// an 84% token-stream divergence on real weights. No numerical test found it: uniform
// random fixtures lack the dynamic range to expose a ~1 ULP data-dependent split, which is
// why aikit's tolerance-based gemv parity test passed throughout.
//
// It reads the COMMITTED artifact, so it needs no NVRTC, no GPU and no rebuild, and runs
// in every job. TestPTXReproducible is what ties that artifact back to the source.
//
// The single mul is deliberate: gemv_w8a8_fwd's two-scale product is an explicit
// __fmul_rn (there is no accumulate to fuse it with).
//
// THE EXPECTED COUNTS ARE VERSION-COUPLED: 13/1 holds at NVRTC 12.9.86, the version every
// committed artifact records and the version TestPTXReproducible pins. A toolchain bump can
// legitimately move them.
//
// THAT IS ALSO THE RISK. The failure mode this gate must survive is not the counts moving —
// it is someone making a red test green by editing the constants. A DROP IN fma.rn.f32 WITH
// A MATCHING RISE IN mul.f32 IS THE SIGNATURE OF THE ORIGINAL DEFECT, not a version
// artifact: it means the compiler chose mul+add where the contract requires fusion, which is
// precisely how gemv_w4a8_fwd shipped at 11 fma + 3 mul against its batched counterpart's
// 192 fma + 0 mul. Re-baselining that away reinstates an 84% token-stream divergence and
// leaves the test green. Before touching these numbers, verify in gemv_quant.cu that every
// float MAC is still an explicit intrinsic — the failure text below says so too, because
// whoever hits it may not read this far.
func TestGemvQuantFMAHistogram(t *testing.T) {
	const (
		path    = "testdata/gemv_quant.ptx"
		wantFMA = 13
		wantMul = 1
		// The toolchain the expectations were taken at; reported on failure so a
		// version bump is distinguishable from a regression at a glance.
		baselineNVRTC = "12.9.86"
	)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if v := ptxToolchainVersion(b); v != "" {
		t.Logf("%s built by NVRTC %s", path, v)
	}
	fma, mul := fmaHistogram(b)
	if fma != wantFMA || mul != wantMul {
		suspect := ""
		if fma < wantFMA && mul > wantMul {
			suspect = "\n\n  *** fma DROPPED and mul ROSE — that is the SIGNATURE OF THE DEFECT this gate " +
				"exists to catch, not a toolchain difference. The compiler chose mul+add where the " +
				"contract requires fusion. Treat this as a regression until proven otherwise. ***"
		}
		t.Errorf("gemv_quant.ptx float mix is %d fma.rn.f32 / %d mul.f32, want %d / %d "+
			"(baseline taken at NVRTC %s; this artifact records %q).%s\n\n"+
			"  DO NOT update these constants to make this pass until you have verified, in "+
			"gemv_quant.cu, that EVERY float MAC is still an explicit intrinsic (__fmaf_rn / "+
			"__fmul_rn / __fadd_rn) — TestKernelFMALint checks exactly that and should be read "+
			"as the first response to this failure. Re-baselining a real regression away is how "+
			"an 84%% token-stream divergence ships with a green test.\n"+
			"  A legitimate cause is an NVRTC version change, which shifts counts without "+
			"changing the source; TestPTXReproducible pins the version precisely so the two "+
			"cases stay distinguishable.",
			fma, mul, wantFMA, wantMul, baselineNVRTC, ptxToolchainVersion(b), suspect)
		return
	}
	t.Logf("gemv_quant.ptx: %d fma.rn.f32 / %d mul.f32 — no compiler discretion remains", fma, mul)
}

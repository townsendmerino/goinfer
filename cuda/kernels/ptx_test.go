//go:build cuda

package kernels

import (
	"fmt"
	"os"
	"path/filepath"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
)

type (
	Device       = gpu.Device
	Buffer       = gpu.Buffer
	Pipeline     = gpu.Pipeline
	LaunchConfig = gpu.LaunchConfig
)

var (
	CreateSystemDefaultDevice = gpu.CreateSystemDefaultDevice
	Arg                       = gpu.Arg
)

// The PTX modules the tests load, by the names cuda/kernels.go gives them.
var (
	argmaxPTX             = readPTX("argmax")
	attnBlockPTX          = readPTX("attn_block")
	attnFusedVitPTX       = readPTX("attn_fused_vit")
	decodeSplitKVPTX      = readPTX("decode_splitkv")
	deltaNetPTX           = readPTX("deltanet")
	fusedQKVPTX           = readPTX("fused_qkv")
	gemvFwdPTX            = readPTX("gemv_fwd")
	geluQuantPTX          = readPTX("gelu_quant")
	gluePTX               = readPTX("glue")
	gptOssActPTX          = readPTX("gptoss_act")
	layernormQuantPTX     = readPTX("layernorm_quant")
	mlaPTX                = readPTX("mla")
	moePTX                = readPTX("moe")
	prefillBatchedKV16PTX = readPTX("prefill_batched_kv16")
	prefillBatchedPTX     = readPTX("prefill_batched")
	prefillSplitKVKV16PTX = readPTX("prefill_splitkv_kv16")
	prefillSplitKVPTX     = readPTX("prefill_splitkv")
	ropeMRopePrefillPTX   = readPTX("rope_mrope_prefill")
	ropePairwisePTX       = readPTX("rope_pairwise")
)

// readPTX reads cuda/testdata/<name>.ptx. The files are committed, so a missing one is a broken checkout and panics
// at init rather than surfacing later as a module-load error inside some test.
func readPTX(name string) []byte {
	b, err := os.ReadFile(filepath.Join("..", "testdata", name+".ptx"))
	if err != nil {
		panic(fmt.Sprintf("cuda/kernels: %v", err))
	}
	return b
}

// moePTXOrOverride returns moe.ptx, or the file GOINFER_MOE_PTX_FILE names: a scratch build at a different MOE_MAX_E,
// so the reservation tests can measure it without regenerating the committed PTX.
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

//go:build cuda

package kernels

import (
	"fmt"
	"os"
	"path/filepath"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
)

var CreateSystemDefaultDevice = gpu.CreateSystemDefaultDevice

// The PTX modules the tests load, by the names cuda/kernels.go gives them.
var (
	argmaxPTX             = readPTX("argmax")
	attnBlockPTX          = readPTX("attn_block")
	decodeSplitKVPTX      = readPTX("decode_splitkv")
	deltaNetPTX           = readPTX("deltanet")
	fusedQKVPTX           = readPTX("fused_qkv")
	gemvFwdPTX            = readPTX("gemv_fwd")
	gluePTX               = readPTX("glue")
	gptOssActPTX          = readPTX("gptoss_act")
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

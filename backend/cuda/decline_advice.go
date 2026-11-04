//go:build cuda

package cuda

import "strings"

// declineAdvice turns the error that made BuildResident decline into the reason an operator reads
// (docs/tasks/task-first-hour.md). The raw error was a trace and a Go internal where the operator needs to know what to
// change (a device OOM or an unpackable weight surfaced as "executor job panicked: ..." plus a goroutine stack, or
// `unsupported projection kind ""`), and the CPU fallback it caused said nothing about how to stay on the GPU.
//
// reason is one line: the cause, then what to try. detail is whatever was cut (the panic's stack, kept because it
// localized a real bug once; see runJob), for stderr only. moeCache says -moe-cache-experts is already on, so it is not
// offered again. An error that is neither a memory failure nor an unpackable weight is a feature decline with its own
// explicit text and is returned as it came.
func declineAdvice(msg string, moeCache bool) (reason, detail string) {
	reason = msg
	if i := strings.Index(msg, "\ngoroutine "); i >= 0 {
		reason, detail = strings.TrimRight(msg[:i], "\n "), msg[i+1:]
	}
	oom := strings.Contains(reason, "CUDA_ERROR_OUT_OF_MEMORY") || strings.Contains(reason, "out of memory")
	emptyWeight := strings.Contains(reason, `unsupported projection kind ""`)
	switch {
	case oom:
		if moeCache {
			return reason + "; the model does not fit this GPU's memory even with -moe-cache-experts: lower -moe-cache-slots or -ctx, " +
				"use a smaller -quant, or run on the CPU on purpose with -backend cpu", detail
		}
		return reason + "; the model does not fit this GPU's memory. Try -moe-cache-experts (a mixture-of-experts model keeps its " +
			"hottest experts on the GPU and streams the rest from host memory), a smaller -quant (int4) or -ctx, " +
			"or run on the CPU on purpose with -backend cpu", detail
	case emptyWeight:
		if moeCache {
			return reason + "; a weight has no packed data the GPU path can use. Load with -quant int4, or run on the CPU on purpose with -backend cpu", detail
		}
		return reason + "; a weight has no packed data the GPU path can use (kind \"\"). On a mixture-of-experts model try " +
			"-moe-cache-experts; otherwise load with -quant int4, or run on the CPU on purpose with -backend cpu", detail
	}
	return reason, detail
}

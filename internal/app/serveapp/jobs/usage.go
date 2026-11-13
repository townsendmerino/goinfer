package jobs

// Usage is a generation's token accounting as the wire carries it: the OpenAI usage block and goinfer's vendor extensions, which
// a Job records.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// PrefillReusedTokens is a goinfer vendor extension (not part of the OpenAI schema; the same convention as
	// pathFields's decode_path and prefill_path on /v1/models): how many leading prompt tokens this generation's
	// prefill skipped because a prior turn left them resident (decoder.Generation.PrefillReused). 0 on a cold
	// prefill, on a path that does not track reuse, or when nothing was reused; the number alone cannot tell those
	// apart. An unrecognised JSON key, so any standard OpenAI client ignores it.
	PrefillReusedTokens int `json:"prefill_reused_tokens"`
	// PrefillReuse is the vendor extension beside it that says where those tokens came from
	// (docs/tasks/task-kv-architecture-2026-10.md §5.16); absent when nothing was reused.
	PrefillReuse *PrefillReuseSources `json:"prefill_reuse,omitempty"`
}

// PrefillReuseSources splits prefill_reused_tokens by source. own_tokens were already in the KV record the request ran on (its own
// continuation, or a previous conversation's matching prefix left in that slot), shared_tokens came from another slot's KV (a copy or a
// shared block), host_tokens a host snapshot restored onto the device, disk_tokens the disk tier. promoted_blocks counts blocks that came back from host memory, the slot's own or a copy's source; the
// slot's own rows count in host_tokens with a host snapshot's, and rows that came through a copy in shared_tokens. The disk tier is KV7-rest's, so disk_tokens is 0 until it lands.
type PrefillReuseSources struct {
	OwnTokens      int `json:"own_tokens"`
	SharedTokens   int `json:"shared_tokens"`
	HostTokens     int `json:"host_tokens"`
	DiskTokens     int `json:"disk_tokens"`
	PromotedBlocks int `json:"promoted_blocks"`
}

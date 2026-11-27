// Package residency implements the resident GPU-decode coordination layer below
// decoder: subsystem no-op defaults and resolution, KV sharing scopes, the
// batched decode token coalescer, resident KV prefix reuse, host KV prefix
// snapshots, and recurrent checkpointing.
package residency

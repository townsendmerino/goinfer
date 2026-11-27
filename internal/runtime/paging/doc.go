// Package paging implements on-demand weight paging over read-only .giw
// mappings: an LRU expert pager (mmap+madvise or owned-buffer pread pool) for
// MoE models and a sliding-window layer pager for dense models.
package paging

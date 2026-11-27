// Package stores holds the bounded in-memory state stores and registries that
// internal/app/serveapp shares across HTTP surfaces:
//
//   - FileStore: OpenAI Files API write-once blobs bounded by count and total bytes
//     (TestFileStore_byteLimitDropsOldestFirst).
//   - BatchStore and BatchRecord: dialect-neutral multi-line batch tracking, server-wide
//     active-line budgeting, and completion assembly over jobs.Store
//     (TestBatchStore_evictionNeverDropsALiveBatch, TestBatchStore_activeLineBudgetIsReservedAndReleased).
//   - GenerationRegistry: process-wide in-flight generation tracking for cancel-by-ID
//     and global halt quiescence (TestGenerationRegistry_cancelAndWaitEmpty).
//   - FeatureCache: SHA-256-keyed LRU cache of encoded vision/audio features bounded
//     by fit-plan leftover bytes (TestFeatureCache_budgetEvictsLeastRecentlyUsed).
//   - ResponseStore: generic FIFO count- and byte-bounded store for /v1/responses
//     previous_response_id continuations (TestResponseStore_byteBudgetEvictsOldestFirst).
//
// It imports no model, tokenizer, chat, or decoder package (rule serveapp-stores-leaf).
package stores

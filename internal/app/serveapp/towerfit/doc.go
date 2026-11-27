// Package towerfit prices device vision towers before resident decoder startup,
// routes vision and audio towers across CPU and GPU backends, checks unified-memory
// WebGPU fit, recovers or falls back to CPU on device memory exhaustion, and
// swaps Gemma 4's device tower against the MoE expert cache.
package towerfit

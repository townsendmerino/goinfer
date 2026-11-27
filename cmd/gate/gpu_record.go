package main

import "github.com/townsendmerino/goinfer/internal/guard/gaterelease"

type (
	gpuRecord       = gaterelease.GPURecord
	heavyProcRecord = gaterelease.HeavyProcRecord
	gpuRecordGroups = gaterelease.GPURecordGroups
)

const gpuRecordDirDefault = gaterelease.GPURecordDirDefault

var (
	runGPURecords  = gaterelease.RunGPURecords
	loadGPURecords = gaterelease.LoadGPURecords
)

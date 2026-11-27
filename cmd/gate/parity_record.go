package main

import "github.com/townsendmerino/goinfer/internal/guard/gaterelease"

type parityRun = gaterelease.ParityRun

const parityRecordDirDefault = gaterelease.ParityRecordDirDefault

var (
	parityQualifyingPaths = gaterelease.ParityQualifyingPaths
	loadParityRecords     = gaterelease.LoadParityRecords
	writeParityRecord     = gaterelease.WriteParityRecord
)

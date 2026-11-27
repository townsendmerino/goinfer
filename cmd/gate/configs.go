package main

import (
	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
	"github.com/townsendmerino/goinfer/internal/guard/gatehygiene"
)

type (
	cell           = gatecore.Cell
	gateConfig     = gatecore.GateConfig
	cellResult     = gatecore.CellResult
	results        = gatecore.Results
	testEvent      = gatecore.TestEvent
	testKey        = gatecore.TestKey
	provenance     = gatecore.Provenance
	procPeak       = gatecore.ProcPeak
	skipRow        = gatecore.SkipRow
	recordCell     = gatecore.RecordCell
	runRecord      = gatecore.RunRecord
	runRecordExtra = gatecore.RunRecordExtra
)

const (
	bold            = gatecore.Bold
	red             = gatecore.Red
	green           = gatecore.Green
	amber           = gatecore.Amber
	off             = gatecore.Off
	runRecordSchema = gatecore.RunRecordSchema
)

var (
	newResults         = gatecore.NewResults
	runCell            = gatecore.RunCell
	sanitize           = gatecore.Sanitize
	censusConfig       = gatecore.CensusConfig
	env                = gatecore.Env
	home               = gatecore.Home
	gatherProvenance   = gatecore.GatherProvenance
	gatherProvenanceIn = gatecore.GatherProvenanceIn
	repoRoot           = gatecore.RepoRoot
	gitOut             = gatecore.GitOut
	gitTree            = gatecore.GitTree
	gitObject          = gatecore.GitObject
	tailLines          = gatecore.TailLines
	writeFileAtomic    = gatecore.WriteFileAtomic
	runRecordPath      = gatecore.RunRecordPath
	writeRunRecord     = gatecore.WriteRunRecord
	loadRunRecord      = gatecore.LoadRunRecord
	loadRunRecordFull  = gatecore.LoadRunRecordFull
	readRunRecord      = gatecore.ReadRunRecord
	startProcSampler   = gatecore.StartProcSampler
	reportCensus       = gatecore.ReportCensus
	reportTally        = gatecore.ReportTally
	gradeCell          = gatecore.GradeCell
	gradeCellError     = gatecore.GradeCellError
	gradeTallyRun      = gatecore.GradeTallyRun
	timeoutPanic       = gatecore.TimeoutPanic
	trunc              = gatecore.Trunc
	isSubtest          = gatecore.IsSubtest
	sortedSet          = gatecore.SortedSet
	orDash             = gatecore.OrDash
	wrapWords          = gatecore.WrapWords
	short8             = gatecore.Short8
	absentBeforeMove   = gatecore.AbsentBeforeMove
	treeAt             = gatecore.TreeAt
	objectAt           = gatecore.ObjectAt
	runChangelog       = gatehygiene.RunChangelog
	newestVersionTag   = gatehygiene.NewestVersionTag
	runCommentsOnly    = gatehygiene.RunCommentsOnly
	replayCommentsOnly = gatehygiene.ReplayCommentsOnly
	runLedger          = gatehygiene.RunLedger
	classifyGate       = gatehygiene.ClassifyGate
	reconcileLedger    = gatehygiene.ReconcileLedger
	runShadow          = gatehygiene.RunShadow
)

func heavyConfig() *gateConfig {
	return gatecore.HeavyConfig()
}

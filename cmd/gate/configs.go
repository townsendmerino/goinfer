package main

import (
	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
	"github.com/townsendmerino/goinfer/internal/guard/gatehygiene"
	"github.com/townsendmerino/goinfer/internal/guard/gatequick"
	"github.com/townsendmerino/goinfer/internal/guard/gaterelease"
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
	heavyConfig        = gatecore.HeavyConfig
	env                = gatecore.Env
	home               = gatecore.Home
	gatherProvenance   = gatecore.GatherProvenance
	repoRoot           = gatecore.RepoRoot
	gitOut             = gatecore.GitOut
	gitTree            = gatecore.GitTree
	gitObject          = gatecore.GitObject
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
	short8             = gatecore.Short8
	absentBeforeMove   = gatecore.AbsentBeforeMove
	treeAt             = gatecore.TreeAt
	objectAt           = gatecore.ObjectAt
	runChangelog       = gatehygiene.RunChangelog
	runCommentsOnly    = gatehygiene.RunCommentsOnly
	replayCommentsOnly = gatehygiene.ReplayCommentsOnly
	runLedger          = gatehygiene.RunLedger
	classifyGate       = gatehygiene.ClassifyGate
	reconcileLedger    = gatehygiene.ReconcileLedger
	runShadow          = gatehygiene.RunShadow
	loadQuickRecords   = gatequick.LoadQuickRecords
	replayRelease      = gaterelease.ReplayRelease
)

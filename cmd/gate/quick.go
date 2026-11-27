package main

import "github.com/townsendmerino/goinfer/internal/guard/gatequick"

const quickFullTimeout = gatequick.QuickFullTimeout

var (
	_           = quickFullTimeout
	runQuick    = gatequick.RunQuick
	replayQuick = gatequick.ReplayQuick
)

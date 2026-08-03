module github.com/townsendmerino/goinfer/cuda

go 1.26.5

require (
	github.com/eitamring/gocudrv v0.3.2
	github.com/townsendmerino/aikit v1.16.0
	github.com/townsendmerino/aikit/gpu v0.21.0
	github.com/townsendmerino/goinfer v0.8.0
)

require (
	github.com/ebitengine/purego v0.10.1 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.40.0 // indirect
)

replace github.com/townsendmerino/goinfer => ../

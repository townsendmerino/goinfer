module github.com/townsendmerino/goinfer/gpu

go 1.26.3

require (
	github.com/cogentcore/webgpu v0.23.0
	github.com/townsendmerino/aikit v1.7.3
	github.com/townsendmerino/goinfer v0.6.0
)

require golang.org/x/text v0.37.0 // indirect

replace github.com/townsendmerino/goinfer => ../

module github.com/townsendmerino/goinfer

go 1.26.3

// Dependencies (golang.org/x/text via tokenizer, github.com/townsendmerino/aikit
// for embed + linalg, and — in the opt-in ./gpu submodule only —
// github.com/cogentcore/webgpu) are added by `go mod tidy` once the decoder /
// tokenizer / constrain packages land here. See aikit/docs/aikit-module-split-plan.md.

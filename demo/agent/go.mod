// demo/agent is a separate module (like ken's chunk/treesitter) so the
// MCP SDK dependency never enters goinfer's dependency-light go.mod.
// Two commands share the agent package: cmd/stdlib-agent (terminal REPL)
// and cmd/agent-web (browser chat).
module github.com/townsendmerino/goinfer/demo/agent

go 1.26.5

require (
	github.com/modelcontextprotocol/go-sdk v1.6.1
	github.com/townsendmerino/aikit v1.16.0
	github.com/townsendmerino/goinfer v0.9.0
	github.com/townsendmerino/goinfer/gpu v0.9.0
)

require (
	github.com/cogentcore/webgpu v0.23.0 // indirect
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/segmentio/asm v1.1.3 // indirect
	github.com/segmentio/encoding v0.5.4 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	golang.org/x/oauth2 v0.35.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.40.0 // indirect
)

// Develop against the checkout; drop this once goinfer tags a release
// that the demo pins.
replace github.com/townsendmerino/goinfer => ../..

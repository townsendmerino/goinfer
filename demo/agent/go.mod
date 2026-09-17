// demo/agent is a separate module (like ken's chunk/treesitter) so the
// MCP SDK dependency never enters goinfer's dependency-light go.mod.
// Two commands share the agent package: cmd/stdlib-agent (terminal REPL)
// and cmd/agent-web (browser chat).
module github.com/townsendmerino/goinfer/demo/agent

go 1.27.0

require (
	github.com/modelcontextprotocol/go-sdk v1.6.1
	github.com/townsendmerino/aikit v1.44.0
	github.com/townsendmerino/goinfer v0.18.0
	github.com/townsendmerino/goinfer/gpu v0.18.0
)

require (
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/oliverbestmann/webgpu v1.36.0 // indirect
	github.com/oliverbestmann/webgpu/libs-android v0.0.0-20260628152806-6b27e30a172e // indirect
	github.com/oliverbestmann/webgpu/libs-darwin v0.0.0-20260628152755-66a5dfa57f8d // indirect
	github.com/oliverbestmann/webgpu/libs-ios v0.0.0-20260628152757-fe2537e7ddac // indirect
	github.com/oliverbestmann/webgpu/libs-linux v0.0.0-20260628152803-421b8a341d08 // indirect
	github.com/oliverbestmann/webgpu/libs-windows v0.0.0-20260628152801-f47d1b682eb8 // indirect
	github.com/segmentio/asm v1.1.3 // indirect
	github.com/segmentio/encoding v0.5.4 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	golang.org/x/oauth2 v0.35.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.40.0 // indirect
)

// Develop against the checkout; drop this once goinfer tags a release
// that the demo pins.

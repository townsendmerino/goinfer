// demo/agent is a separate module (like ken's chunk/treesitter) so the
// MCP SDK dependency never enters goinfer's dependency-light go.mod.
// Two commands share the agent package: cmd/stdlib-agent (terminal REPL)
// and cmd/agent-web (browser chat).
module github.com/townsendmerino/goinfer/demo/agent

go 1.27.0

require (
	github.com/modelcontextprotocol/go-sdk v1.6.1
	github.com/townsendmerino/goinfer v0.24.1
	github.com/townsendmerino/goinfer/backend/webgpu v0.24.1
)

require (
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/oliverbestmann/webgpu v1.38.0 // indirect
	github.com/oliverbestmann/webgpu/libs-android v0.0.0-20261112101134-9fc0777bdf01 // indirect
	github.com/oliverbestmann/webgpu/libs-darwin v0.0.0-20261112101121-f12e930cb707 // indirect
	github.com/oliverbestmann/webgpu/libs-ios v0.0.0-20261112101124-0a18a0d4d894 // indirect
	github.com/oliverbestmann/webgpu/libs-linux v0.0.0-20261112101129-823eab7d2391 // indirect
	github.com/oliverbestmann/webgpu/libs-windows v0.0.0-20261112110423-e4fd7225cf83 // indirect
	github.com/segmentio/asm v1.1.3 // indirect
	github.com/segmentio/encoding v0.5.4 // indirect
	github.com/townsendmerino/aikit v1.76.0 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	golang.org/x/oauth2 v0.35.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.40.0 // indirect
)

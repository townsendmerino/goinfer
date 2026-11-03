// sdktest is a separate module so the SDKs it drives never enter goinfer's go.mod, which lists two requirements.
// It holds only tests: the official OpenAI and Anthropic Go SDKs against the server served in process
// (package server). It is not tagged or published.
module github.com/townsendmerino/goinfer/sdktest

go 1.27.0

require (
	github.com/anthropics/anthropic-sdk-go v1.80.0
	github.com/openai/openai-go/v3 v3.77.0
)

require (
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.2 // indirect
	github.com/coder/websocket v1.8.15 // indirect
	github.com/invopop/jsonschema v0.14.0 // indirect
	github.com/pb33f/ordered-map/v2 v2.3.1 // indirect
	github.com/standard-webhooks/standard-webhooks/libraries v0.0.1 // indirect
	github.com/tidwall/gjson v1.19.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/townsendmerino/goinfer v0.22.0
	go.yaml.in/yaml/v4 v4.0.0-rc.2 // indirect
	golang.org/x/sync v0.23.0 // indirect
)

//go:build tools

// Package tools pins the staticcheck CI builds (action.yml): v0.8.1 against golang.org/x/tools v0.51.0. No staticcheck
// release reads Go 1.27.2's export data (version 5; v0.8.0 and v0.8.1 both stop at 4), and golang.org/x/tools v0.51.0's
// importer does, so CI builds this pair from source instead of downloading the release binary (owner, 2026-10-08:
// "lets move to 1.27.2").
package tools

import _ "honnef.co/go/tools/cmd/staticcheck"

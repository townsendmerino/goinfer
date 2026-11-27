package serveapp

import "github.com/townsendmerino/goinfer/internal/app/serveapp/adminsock"

var (
	isAdminCLICmd = adminsock.IsCLICmd
	runAdminCLI   = adminsock.RunCLI
)

package serveapp

import (
	"net/http"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/adminsock"
)

var defaultAdminSocketPath = adminsock.DefaultSocketPath

// startAdminSocket mounts every /admin/* route onto a 0600 Unix socket with no
// API-key check (file permissions gate access).
func startAdminSocket(s *server, path string, textCap int64) (close func(), err error) {
	mux := http.NewServeMux()
	registerAdminRoutes(mux, s, textCap, func(h http.HandlerFunc) http.HandlerFunc { return h })
	return adminsock.Start(mux, path)
}

package serveapp

import (
	"net"
	"net/http"
	"strings"
)

// hostGuard refuses a request whose Host header does not name this server, for a listener bound to the loopback only.
//
// DNS rebinding: a page the user has open resolves evil.example to 127.0.0.1 after its first lookup, so the browser sends
// requests to this server with Host and Origin both evil.example:8080. Origin equals Host, so sameOrigin lets them through, and the
// page, same-origin by the browser's own reckoning, can read every response. The Host header is the one thing the page cannot
// choose, so a loopback listener answers only the names that reach loopback by themselves (localhost, 127.0.0.1, ::1), the host
// it was told to bind, and any -allowed-hosts names the operator lists (a reverse proxy on the same machine forwards the public name).
//
// A listener bound beyond loopback is not guarded: its clients legitimately arrive under any name, and the key protects it.
// A request with no Host at all (HTTP/1.0, a raw client) is let through: a browser always sends one.
func hostGuard(next http.Handler, bindAddr string, extra []string) http.Handler {
	allowed := map[string]bool{"localhost": true, "127.0.0.1": true, "::1": true}
	if h, _, err := net.SplitHostPort(bindAddr); err == nil && h != "" {
		allowed[normalizeHost(h)] = true
	}
	for _, h := range extra {
		if h = normalizeHost(h); h != "" {
			allowed[h] = true
		}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Host != "" && !allowed[normalizeHost(r.Host)] {
			// 421 Misdirected Request: the request reached a server that will not answer under this name. The refusal names
			// the flag, so a reverse-proxy operator sees what to set.
			writeErr(w, http.StatusMisdirectedRequest, "unrecognised Host header; this server is bound to the loopback and answers only localhost, 127.0.0.1 and [::1] (add a name with -allowed-hosts)")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// normalizeHost reduces a Host header value or a -allowed-hosts entry to a bare lower-case host: no port, no IPv6 brackets, no trailing dot.
func normalizeHost(h string) string {
	h = strings.TrimSpace(h)
	if host, port, err := net.SplitHostPort(h); err == nil && isDigits(port) { // a port is digits: "localhost:80@evil.example" is not localhost
		h = host
	}
	h = strings.TrimSuffix(strings.TrimPrefix(h, "["), "]")
	return strings.ToLower(strings.TrimSuffix(h, "."))
}

// splitHosts splits a comma-separated -allowed-hosts value.
func splitHosts(s string) []string {
	var out []string
	for _, h := range strings.Split(s, ",") {
		if h = strings.TrimSpace(h); h != "" {
			out = append(out, h)
		}
	}
	return out
}

func isDigits(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return s != ""
}

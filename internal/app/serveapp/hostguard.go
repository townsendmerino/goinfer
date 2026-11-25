package serveapp

import (
	"maps"
	"net"
	"net/http"
	"net/url"
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
	maps.Copy(allowed, hostSet(extra))
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

// serveHandler wraps mux in the guards of a listener bound to addr: on loopback, hostGuard and, with no key, browserGuard inside
// it; beyond loopback, mux unchanged (the key protects it).
func serveHandler(mux http.Handler, addr, authKey, allowedHosts string) http.Handler {
	if !addrIsLoopback(addr) {
		return mux
	}
	hosts := splitHosts(allowedHosts)
	return hostGuard(browserGuard(mux, authKey, hosts), addr, hosts)
}

// browserGuard refuses, with 403, a POST, PUT, PATCH or DELETE under /v1/ that a browser sent cross-origin (crossOriginBrowser,
// with proxied the -allowed-hosts names). With no key requireAuth is a no-op, and any page open in the user's browser can send
// such a request (the browser blocks reading the reply, not sending it), so on a key-free loopback server it would start
// generations, uploads and batches. A request with neither header (curl, the SDKs) passes, as does a same-origin one. With authKey
// set it returns next unchanged.
func browserGuard(next http.Handler, authKey string, proxied []string) http.Handler {
	if authKey != "" {
		return next
	}
	names := hostSet(proxied)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
			if strings.HasPrefix(r.URL.Path, "/v1/") && crossOriginBrowser(r, names) {
				writeErr(w, http.StatusForbidden, "cross-origin browser request refused: this loopback server has no -api-key, so it acts only on same-origin requests")
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// crossOriginBrowser reports whether r carries browser evidence of coming from another origin: Sec-Fetch-Site is cross-site, or
// Origin is unparsable, "null", or names another host and port than Host, unless its host is in proxied (normalizeHost form). A
// reverse proxy may rewrite Host to the loopback address, so a page it serves under a name the operator listed in -allowed-hosts
// arrives with that name in Origin and the loopback in Host; such a name matches on any port. A page cannot choose its own
// Origin, so a listed name admits only pages served under it.
func crossOriginBrowser(r *http.Request, proxied map[string]bool) bool {
	if strings.EqualFold(r.Header.Get("Sec-Fetch-Site"), "cross-site") {
		return true
	}
	o := r.Header.Get("Origin")
	if o == "" {
		return false
	}
	u, err := url.Parse(o)
	if err != nil || o == "null" || u.Host == "" {
		return true
	}
	if proxied[normalizeHost(u.Host)] {
		return false
	}
	def := "80"
	if u.Scheme == "https" {
		def = "443"
	}
	return canonicalHostPort(u.Host, def) != canonicalHostPort(r.Host, def)
}

// hostSet is hosts in normalizeHost form, as a set; empty names are dropped.
func hostSet(hosts []string) map[string]bool {
	set := make(map[string]bool, len(hosts))
	for _, h := range hosts {
		if h = normalizeHost(h); h != "" {
			set[h] = true
		}
	}
	return set
}

// canonicalHostPort reduces a host[:port] to lower case with no IPv6 brackets and no trailing dot, dropping the port when it is
// defaultPort, so an Origin and a Host header naming the same server compare equal.
func canonicalHostPort(hp, defaultPort string) string {
	host, port := hp, ""
	if h, p, err := net.SplitHostPort(hp); err == nil {
		host, port = h, p
	}
	host = strings.ToLower(strings.TrimSuffix(strings.TrimSuffix(strings.TrimPrefix(host, "["), "]"), "."))
	if port == "" || port == defaultPort {
		return host
	}
	return net.JoinHostPort(host, port)
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
	for h := range strings.SplitSeq(s, ",") {
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

package netguard

import (
	"maps"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// AddrIsLoopback reports whether addr binds exclusively to the loopback interface.
func AddrIsLoopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}
	if host == "" {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// HostGuard refuses requests whose Host header is outside localhost, 127.0.0.1,
// ::1, bindAddr's host, and extra, preventing DNS-rebinding attacks against a
// loopback listener.
func HostGuard(next http.Handler, bindAddr string, extra []string) http.Handler {
	allowed := map[string]bool{"localhost": true, "127.0.0.1": true, "::1": true}
	if h, _, err := net.SplitHostPort(bindAddr); err == nil && h != "" {
		allowed[NormalizeHost(h)] = true
	}
	maps.Copy(allowed, HostSet(extra))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Host != "" && !allowed[NormalizeHost(r.Host)] {
			WriteErr(w, http.StatusMisdirectedRequest, "unrecognised Host header; this server is bound to the loopback and answers only localhost, 127.0.0.1 and [::1] (add a name with -allowed-hosts)")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ServeHandler wraps mux in HostGuard and (when authKey is empty) BrowserGuard
// when addr binds to loopback, and returns mux unchanged otherwise.
func ServeHandler(mux http.Handler, addr, authKey, allowedHosts string) http.Handler {
	if !AddrIsLoopback(addr) {
		return mux
	}
	hosts := SplitHosts(allowedHosts)
	return HostGuard(BrowserGuard(mux, authKey, hosts), addr, hosts)
}

// BrowserGuard refuses cross-origin mutating requests under /v1/ on a key-free
// loopback server with 403 Forbidden.
func BrowserGuard(next http.Handler, authKey string, proxied []string) http.Handler {
	if authKey != "" {
		return next
	}
	names := HostSet(proxied)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
			if strings.HasPrefix(r.URL.Path, "/v1/") && CrossOriginBrowser(r, names) {
				WriteErr(w, http.StatusForbidden, "cross-origin browser request refused: this loopback server has no -api-key, so it acts only on same-origin requests")
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// SameOriginGuard wraps web-UI handlers so cross-origin browser requests are
// refused with 403 Forbidden unless Origin matches Host or proxied.
func SameOriginGuard(proxied []string) func(http.HandlerFunc) http.HandlerFunc {
	names := HostSet(proxied)
	return func(h http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if CrossOriginBrowser(r, names) {
				WriteErr(w, http.StatusForbidden, "cross-origin request refused")
				return
			}
			h(w, r)
		}
	}
}

// CrossOriginBrowser reports whether r carries browser headers indicating a cross-origin request.
func CrossOriginBrowser(r *http.Request, proxied map[string]bool) bool {
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
	if proxied[NormalizeHost(u.Host)] {
		return false
	}
	def := "80"
	if u.Scheme == "https" {
		def = "443"
	}
	return CanonicalHostPort(u.Host, def) != CanonicalHostPort(r.Host, def)
}

// HostSet normalizes hosts into a lookup set, dropping empty entries.
func HostSet(hosts []string) map[string]bool {
	set := make(map[string]bool, len(hosts))
	for _, h := range hosts {
		if h = NormalizeHost(h); h != "" {
			set[h] = true
		}
	}
	return set
}

// CanonicalHostPort lowers host[:port], strips IPv6 brackets and trailing dots,
// and elides defaultPort.
func CanonicalHostPort(hp, defaultPort string) string {
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

// NormalizeHost reduces a Host header or -allowed-hosts entry to a bare lower-case host.
func NormalizeHost(h string) string {
	h = strings.TrimSpace(h)
	if host, port, err := net.SplitHostPort(h); err == nil && isDigits(port) {
		h = host
	}
	h = strings.TrimSuffix(strings.TrimPrefix(h, "["), "]")
	return strings.ToLower(strings.TrimSuffix(h, "."))
}

// SplitHosts splits a comma-separated -allowed-hosts string.
func SplitHosts(s string) []string {
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

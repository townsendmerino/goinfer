package netguard

import (
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHostGuard_refusesRebindingNamesOnALoopbackListener(t *testing.T) {
	reached := 0
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reached++; w.WriteHeader(http.StatusOK) })
	h := HostGuard(inner, "127.0.0.1:8080", []string{"goinfer.lan", " Proxy.Example.com. "})

	for _, c := range []struct {
		host string
		want int
	}{
		{"127.0.0.1:8080", 200}, {"127.0.0.1", 200}, {"localhost:8080", 200}, {"LocalHost", 200}, {"[::1]:8080", 200}, {"[::1]", 200},
		{"goinfer.lan:8080", 200}, {"proxy.example.com", 200},
		{"", 200},
		{"evil.example:8080", 421}, {"evil.example", 421}, {"127.0.0.1.evil.example:8080", 421}, {"localhost.evil.example", 421},
		{"192.168.1.5:8080", 421}, {"0.0.0.0:8080", 421}, {"[::2]:8080", 421}, {"localhost:8080@evil.example", 421},
	} {
		r := httptest.NewRequest("GET", "/v1/models", nil)
		r.Host = c.host
		r.Header.Set("Origin", "http://"+c.host)
		w := httptest.NewRecorder()
		before := reached
		h.ServeHTTP(w, r)
		if w.Code != c.want {
			t.Errorf("Host %q: status %d, want %d", c.host, w.Code, c.want)
		}
		if c.want == 421 {
			if reached != before {
				t.Errorf("Host %q: handler ran for refused request", c.host)
			}
			if !strings.Contains(w.Body.String(), "allowed-hosts") {
				t.Errorf("Host %q: refusal missing flag hint: %s", c.host, w.Body.String())
			}
		}
	}
}

func TestHostGuard_realSocketRebinding(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skip(err)
	}
	srv := &http.Server{Handler: HostGuard(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("secret")) }), ln.Addr().String(), nil)}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()
	_, port, _ := net.SplitHostPort(ln.Addr().String())
	for host, want := range map[string]int{ln.Addr().String(): 200, "evil.example:" + port: 421} {
		req, _ := http.NewRequest("GET", "http://"+ln.Addr().String()+"/", nil)
		req.Host = host
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != want {
			t.Errorf("Host %q: %d, want %d", host, resp.StatusCode, want)
		}
	}
}

func TestHostGuard_predicateAndBrowserGuard(t *testing.T) {
	for addr, loop := range map[string]bool{"127.0.0.1:8080": true, "localhost:8080": true, "[::1]:8080": true, "0.0.0.0:8080": false, ":8080": false, "10.0.0.5:8080": false} {
		if got := AddrIsLoopback(addr); got != loop {
			t.Errorf("AddrIsLoopback(%q) = %v, want %v", addr, got, loop)
		}
	}
	if got := SplitHosts(" a.example , ,b.example,"); len(got) != 2 || got[0] != "a.example" || got[1] != "b.example" {
		t.Errorf("SplitHosts: %v", got)
	}
	const addr = "127.0.0.1:8080"
	reached := 0
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reached++; w.WriteHeader(http.StatusOK) })
	for _, c := range []struct {
		name, addr, key, method, path, host, origin, fetchSite string
		want                                                   int
		allowed                                                string
	}{
		{"cross-origin Origin", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "http://evil.example", "", 403, ""},
		{"another local port", addr, "", "POST", "/v1/chat/completions", "localhost:8080", "http://localhost:3000", "", 403, ""},
		{"Origin null", addr, "", "POST", "/v1/messages", "127.0.0.1:8080", "null", "", 403, ""},
		{"unparsable Origin", addr, "", "POST", "/v1/messages", "127.0.0.1:8080", "http://[::1", "", 403, ""},
		{"Sec-Fetch-Site cross-site", addr, "", "POST", "/v1/files", "127.0.0.1:8080", "", "cross-site", 403, ""},
		{"DELETE cross-origin", addr, "", "DELETE", "/v1/jobs/x", "127.0.0.1:8080", "http://evil.example", "", 403, ""},
		{"same-origin Origin", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "http://127.0.0.1:8080", "same-origin", 200, ""},
		{"same-origin default port", "localhost:80", "", "POST", "/v1/chat/completions", "LOCALHOST", "http://localhost:80", "", 200, ""},
		{"same-origin IPv6", "[::1]:8080", "", "POST", "/v1/chat/completions", "[::1]:8080", "http://[::1]:8080", "", 200, ""},
		{"absent Origin", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "", "", 200, ""},
		{"GET cross-origin", addr, "", "GET", "/v1/models", "127.0.0.1:8080", "http://evil.example", "cross-site", 200, ""},
		{"keyed server", addr, "k", "POST", "/v1/chat/completions", "127.0.0.1:8080", "http://evil.example", "cross-site", 200, ""},
		{"non-loopback server", "0.0.0.0:8080", "", "POST", "/v1/chat/completions", "10.0.0.5:8080", "http://evil.example", "cross-site", 200, ""},
		{"proxied page allowed", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "https://llm.example.com", "same-origin", 200, "llm.example.com"},
		{"proxied page unlisted", addr, "", "POST", "/v1/chat/completions", "127.0.0.1:8080", "https://llm.example.com", "same-origin", 403, ""},
	} {
		h := ServeHandler(inner, c.addr, c.key, c.allowed)
		r := httptest.NewRequest(c.method, c.path, nil)
		r.Host = c.host
		if c.origin != "" {
			r.Header.Set("Origin", c.origin)
		}
		if c.fetchSite != "" {
			r.Header.Set("Sec-Fetch-Site", c.fetchSite)
		}
		w := httptest.NewRecorder()
		before := reached
		h.ServeHTTP(w, r)
		if w.Code != c.want {
			t.Errorf("%s: status %d, want %d (%s)", c.name, w.Code, c.want, w.Body.String())
		}
		if c.want == 403 && reached != before {
			t.Errorf("%s: handler ran for refused request", c.name)
		}
	}
}

func TestAuthAndMaxBytesMiddleware(t *testing.T) {
	ok := func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }
	auth := RequireAuth("secret-key", ok)

	r1 := httptest.NewRequest("GET", "/v1/models", nil)
	w1 := httptest.NewRecorder()
	auth(w1, r1)
	if w1.Code != http.StatusUnauthorized {
		t.Errorf("missing key status = %d, want 401", w1.Code)
	}

	r2 := httptest.NewRequest("GET", "/v1/models", nil)
	r2.Header.Set("Authorization", "Bearer secret-key")
	w2 := httptest.NewRecorder()
	auth(w2, r2)
	if w2.Code != http.StatusOK {
		t.Errorf("valid bearer status = %d, want 200", w2.Code)
	}

	capped := MaxBytes(4, ok)
	r3 := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("12345"))
	w3 := httptest.NewRecorder()
	capped(w3, r3)
	if w3.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("oversized body status = %d, want 413", w3.Code)
	}
}

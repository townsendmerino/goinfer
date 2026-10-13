package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// A DNS-rebinding request names the attacker's host in Host and Origin alike, which is what defeats sameOrigin; hostGuard refuses it.
func TestHostGuard_refusesRebindingNames(t *testing.T) {
	h := hostGuard(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }), "127.0.0.1:8484")
	for host, want := range map[string]int{
		"127.0.0.1:8484": 200, "localhost:8484": 200, "[::1]:8484": 200, "": 200,
		"evil.example:8484": 421, "evil.example": 421, "localhost:80@evil.example": 421, "127.0.0.1.evil.example": 421,
	} {
		r := httptest.NewRequest("POST", "/api/chat", nil)
		r.Host = host
		r.Header.Set("Origin", "http://"+host)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != want {
			t.Errorf("Host %q: %d, want %d", host, w.Code, want)
		}
	}
	for addr, want := range map[string]bool{"127.0.0.1:8484": true, "localhost:8484": true, "0.0.0.0:8484": false, ":8484": false} {
		if got := addrIsLoopback(addr); got != want {
			t.Errorf("addrIsLoopback(%q) = %v", addr, got)
		}
	}
}

package pull

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func mustReq(t *testing.T, raw string) *http.Request {
	t.Helper()
	r, err := http.NewRequest(http.MethodGet, raw, nil)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestRedirectPolicy_refusesDowngradeAndInternalTargets drives redirectPolicy directly: a public origin cannot be
// redirected to plain HTTP or to a loopback, private, link-local or unspecified address, while loopback-to-loopback
// (an httptest server, a local mirror) and public-to-public redirects still follow.
func TestRedirectPolicy_refusesDowngradeAndInternalTargets(t *testing.T) {
	const public = "https://203.0.113.10/o/r/resolve/main/m.gguf"
	cases := []struct {
		name, from, to string
		refuse         bool
	}{
		{"https to http", public, "http://203.0.113.11/m.gguf", true},
		{"https to http on loopback origin", "https://127.0.0.1:8443/x", "http://127.0.0.1:8080/x", true},
		{"public to loopback v4", public, "https://127.0.0.1/x", true},
		{"public to loopback v6", public, "https://[::1]/x", true},
		{"public to v4-mapped loopback", public, "https://[::ffff:127.0.0.1]/x", true},
		{"public to 10/8", public, "https://10.0.0.1/x", true},
		{"public to 172.16/12", public, "https://172.16.5.4/x", true},
		{"public to 192.168/16", public, "https://192.168.1.1/x", true},
		{"public to ULA", public, "https://[fd00::1]/x", true},
		{"public to metadata link-local", public, "https://169.254.169.254/latest", true},
		{"public to v6 link-local", public, "https://[fe80::1]/x", true},
		{"public to link-local multicast", public, "https://224.0.0.251/x", true},
		{"public to unspecified", public, "https://0.0.0.0/x", true},
		{"public to localhost name", public, "https://localhost/x", true},
		{"public to CGNAT 100.64/10", public, "https://100.100.1.1/x", true},
		{"public to this-network 0/8", public, "https://0.1.2.3/x", true},
		{"public to benchmarking 198.18/15", public, "https://198.19.0.1/x", true},
		{"public to global multicast", public, "https://239.1.2.3/x", true},
		{"public to v6 multicast", public, "https://[ff0e::1]/x", true},
		{"public to broadcast", public, "https://255.255.255.255/x", true},
		{"public to unresolvable name", public, "https://nonexistent.invalid/x", true},
		{"unresolvable origin to loopback", "https://nonexistent.invalid/a", "https://127.0.0.1/x", true},
		{"public to public", public, "https://203.0.113.11/m.gguf", false},
		{"loopback to loopback", "http://127.0.0.1:8080/a", "http://127.0.0.1:8081/b", false},
		{"private to private", "https://10.0.0.1/a", "https://10.0.0.2/b", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := redirectPolicy(nil)(mustReq(t, c.to), []*http.Request{mustReq(t, c.from)})
			if c.refuse && err == nil {
				t.Fatalf("redirect %s -> %s was allowed", c.from, c.to)
			}
			if !c.refuse && err != nil {
				t.Fatalf("redirect %s -> %s refused: %v", c.from, c.to, err)
			}
		})
	}
	// The hop checked is the one immediately before the target: a public hop anywhere in the chain cannot hand off to
	// an internal address.
	via := []*http.Request{mustReq(t, "http://127.0.0.1/a"), mustReq(t, "http://203.0.113.10/b")}
	if err := redirectPolicy(nil)(mustReq(t, "http://127.0.0.1/c"), via); err == nil {
		t.Fatal("loopback -> public -> loopback was allowed")
	}
	var ten []*http.Request
	for range 10 {
		ten = append(ten, mustReq(t, public))
	}
	if err := redirectPolicy(nil)(mustReq(t, public), ten); err == nil {
		t.Fatal("an 11th redirect was followed")
	}
}

// TestRedirectPolicy_proxy: behind a proxy, a target name this host cannot resolve is left to the proxy, while a
// literal internal address is still refused, and a source that does not resolve still has its target judged.
func TestRedirectPolicy_proxy(t *testing.T) {
	proxyURL, _ := url.Parse("http://proxy.example:3128")
	viaProxy := redirectPolicy(func(*http.Request) (*url.URL, error) { return proxyURL, nil })
	noProxy := redirectPolicy(func(*http.Request) (*url.URL, error) { return nil, nil })
	from := []*http.Request{mustReq(t, "https://huggingface.invalid/o/r/resolve/main/m.gguf")}
	if err := viaProxy(mustReq(t, "https://cdn.invalid/m.gguf"), from); err != nil {
		t.Fatalf("an unresolvable target behind a proxy was refused: %v", err)
	}
	if err := viaProxy(mustReq(t, "https://169.254.169.254/latest"), from); err == nil {
		t.Fatal("an unresolvable source behind a proxy redirected to a link-local address")
	}
	if err := viaProxy(mustReq(t, "https://10.0.0.1/x"), from); err == nil {
		t.Fatal("an unresolvable source behind a proxy redirected to a private address")
	}
	if err := noProxy(mustReq(t, "https://cdn.invalid/m.gguf"), from); err == nil {
		t.Fatal("an unresolvable target with no proxy was followed")
	}
}

// TestDownload_redirectDowngradeRefused proves the production client carries the redirect policy: a TLS origin that
// redirects to a plain-HTTP host fails the download before the plain host is contacted.
func TestDownload_redirectDowngradeRefused(t *testing.T) {
	var plainHits atomic.Int32
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		plainHits.Add(1)
		_, _ = w.Write([]byte("payload"))
	}))
	defer plain.Close()
	tlsSrv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, plain.URL+r.URL.Path, http.StatusFound)
	}))
	defer tlsSrv.Close()
	orig := hfCDN
	hfCDN = tlsSrv.URL
	t.Cleanup(func() { hfCDN = orig })

	c := newHTTPClient()
	c.Transport.(*http.Transport).TLSClientConfig = tlsSrv.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	ctx := WithHTTPClient(context.Background(), c)
	dir := t.TempDir()
	if _, err := Download(ctx, "o/r", File{Path: "m.gguf", Size: 7}, dir, nil); err == nil {
		t.Fatal("an https -> http redirect was followed")
	}
	if n := plainHits.Load(); n != 0 {
		t.Fatalf("the plain-HTTP target was contacted %d time(s)", n)
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf")); err == nil {
		t.Fatal("a file was published from a downgraded redirect")
	}
}

func TestSafeRepoPath(t *testing.T) {
	for _, c := range []struct {
		p  string
		ok bool
	}{
		{"model.gguf", true},
		{"Q8_0/model-00001-of-00002.gguf", true},
		{"vision/model.safetensors", true},
		{"", false},
		{"..", false},
		{"../x.gguf", false},
		{"a/../b.gguf", false},
		{"a/..", false},
		{"./x.gguf", false},
		{"a//b.gguf", false},
		{"/etc/passwd", false},
		{"a\\..\\b.gguf", false},
		{"a\x00b.gguf", false},
		{"x.gguf\x00", false},
	} {
		if got := safeRepoPath(c.p); got != c.ok {
			t.Errorf("safeRepoPath(%q) = %v, want %v", c.p, got, c.ok)
		}
	}
}

// TestList_refusesUnsafePathFromServer: a listing naming a traversal path is an error, not a file offered to Download.
func TestList_refusesUnsafePathFromServer(t *testing.T) {
	withMockHF(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"type":"file","path":"ok.gguf","size":1},{"type":"file","path":"../../evil.gguf","size":1}]`))
	})
	files, err := List(context.Background(), "o/r")
	if err == nil {
		t.Fatalf("List accepted an unsafe path: %+v", files)
	}
	if !strings.Contains(err.Error(), "evil.gguf") {
		t.Errorf("error does not name the path: %v", err)
	}
}

// TestDownload_refusesUnsafePath: Download makes no request for a path that could escape the resolve URL or dir.
func TestDownload_refusesUnsafePath(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = w.Write([]byte("x"))
	}))
	defer srv.Close()
	orig := hfCDN
	hfCDN = srv.URL
	t.Cleanup(func() { hfCDN = orig })
	for _, p := range []string{"../x.gguf", "a/../../x.gguf", "/abs.gguf", "a\\b.gguf", "a\x00.gguf", ""} {
		if _, err := Download(context.Background(), "o/r", File{Path: p, Size: 1}, t.TempDir(), nil); err == nil {
			t.Errorf("Download(%q) succeeded", p)
		}
	}
	if n := hits.Load(); n != 0 {
		t.Fatalf("Download made %d request(s) for unsafe paths", n)
	}
}

// TestMetadata_oversizedBodyRefused: a metadata response over maxPullMetaBytes is an error for every JSON decode
// site, not a decode of whatever prefix happened to parse.
func TestMetadata_oversizedBodyRefused(t *testing.T) {
	pad := strings.Repeat("x", maxPullMetaBytes+1)
	cases := []struct {
		name string
		body string
		call func() error
	}{
		{"CheckAccess", `{"gated":false,"private":false,"pad":"` + pad + `"}`, func() error {
			return CheckAccess(context.Background(), "o/r")
		}},
		{"List", `[{"type":"directory","path":"` + pad + `"}]`, func() error {
			_, err := List(context.Background(), "o/r")
			return err
		}},
		{"listTree", `[{"type":"directory","path":"` + pad + `"}]`, func() error {
			_, err := listTree(context.Background(), "o/r")
			return err
		}},
		{"Search", `[{"id":"` + pad + `","private":true}]`, func() error {
			_, err := Search(context.Background(), "q", "gguf", 0)
			return err
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			withMockHF(t, func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(c.body)) })
			err := c.call()
			if err == nil {
				t.Fatalf("%s accepted a %d-byte metadata body", c.name, len(c.body))
			}
			if !strings.Contains(err.Error(), "larger than") {
				t.Fatalf("%s failed for another reason: %v", c.name, err)
			}
		})
	}
}

// TestDownload_overshootRefused: a body longer than the declared size fails and publishes nothing, even with no digest
// to catch it.
func TestDownload_overshootRefused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("a", 200)))
	}))
	defer srv.Close()
	orig := hfCDN
	hfCDN = srv.URL
	t.Cleanup(func() { hfCDN = orig })
	dir := t.TempDir()
	_, err := Download(context.Background(), "o/r", File{Path: "m.gguf", Size: 100}, dir, nil)
	if err == nil {
		t.Fatal("a 200-byte body for a 100-byte file was accepted")
	}
	if !strings.Contains(err.Error(), "more than") {
		t.Fatalf("failed for another reason: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf")); err == nil {
		t.Fatal("an oversized body was published at the final path")
	}
	if st, err := os.Stat(filepath.Join(dir, "m.gguf.part")); err == nil && st.Size() > 100 {
		t.Fatalf("a %d-byte .part was kept for a 100-byte file", st.Size())
	}
}

// TestDownload_shortBodyRefusedAndKept: a body that ends cleanly short of the declared size, with no digest to catch
// it, is an error, publishes nothing at the final path, and keeps the .part so a re-run resumes it.
func TestDownload_shortBodyRefusedAndKept(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("a", 50)))
	}))
	defer srv.Close()
	orig := hfCDN
	hfCDN = srv.URL
	t.Cleanup(func() { hfCDN = orig })
	dir := t.TempDir()
	_, err := Download(context.Background(), "o/r", File{Path: "m.gguf", Size: 100}, dir, nil)
	if err == nil {
		t.Fatal("a 50-byte body for a 100-byte file was accepted")
	}
	if !strings.Contains(err.Error(), "ended at") {
		t.Fatalf("failed for another reason: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf")); err == nil {
		t.Fatal("a short body was published at the final path")
	}
	st, err := os.Stat(filepath.Join(dir, "m.gguf.part"))
	if err != nil || st.Size() != 50 {
		t.Fatalf("the 50-byte .part was not kept for resume: %v %v", st, err)
	}
}

package serveapp

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/inproc"
)

// uploadBody is a multipart/form-data body with one file part.
func uploadBody(t *testing.T, data []byte) ([]byte, string) {
	t.Helper()
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	fw, err := mw.CreateFormFile("file", "a.bin")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fw.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes(), mw.FormDataContentType()
}

// removeTempFiles empties dir and names what it held.
func removeTempFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
		if err := os.RemoveAll(filepath.Join(dir, e.Name())); err != nil {
			t.Fatal(err)
		}
	}
	return names
}

// An upload over ParseMultipartForm's memory spills to a temp file. net/http removes a form only when it was parsed on
// the request the server made, and traceRequests and NewHandler hand the handler a copy, so each route is taken here as
// it is registered: the binary's route table over a listener, and the library's handler over a listener and in process.
func TestUploads_leaveNoTempFiles(t *testing.T) {
	body, ctype := uploadBody(t, make([]byte, maxFileUploadMemory+1<<20))

	srv, err := newServer(config{})
	if err != nil {
		t.Fatal(err)
	}
	binary, _ := srv.routes(config{}, "", nil)
	library, closer, err := NewHandler([]string{"-model", "tiny=" + tinyChatCheckpoint(t), "-backend", "cpu", "-quant", "int8int8"})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	defer closer.Close()

	dir := t.TempDir()
	for _, env := range []string{"TMPDIR", "TMP", "TEMP"} {
		t.Setenv(env, dir)
	}

	// The control: this body does spill, or every case below would pass on nothing.
	spilled := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	spilled.Header.Set("Content-Type", ctype)
	if err := spilled.ParseMultipartForm(maxFileUploadMemory); err != nil {
		t.Fatal(err)
	}
	if got := removeTempFiles(t, dir); len(got) == 0 {
		t.Fatalf("a %d-byte upload left nothing in the temp directory, so this test proves nothing", len(body))
	}

	overListener := func(h http.Handler) func(path string) int {
		return func(path string) int {
			ts := httptest.NewServer(h)
			defer ts.Close() // waits for the handler and for net/http's own cleanup
			resp, err := http.Post(ts.URL+path, ctype, bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			_, _ = io.Copy(io.Discard, resp.Body)
			return resp.StatusCode
		}
	}
	inProcess := func(h http.Handler) func(path string) int {
		return func(path string) int {
			c := &http.Client{Transport: inproc.Transport(h)}
			resp, err := c.Post("http://goinfer"+path, ctype, bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			_, _ = io.Copy(io.Discard, resp.Body) // to EOF: the handler has returned
			return resp.StatusCode
		}
	}
	for name, post := range map[string]func(path string) int{
		"binary over a listener":  overListener(binary),
		"library over a listener": overListener(library),
		"library in process":      inProcess(library),
	} {
		// /v1/files stores the upload; /v1/audio/transcriptions refuses it, with no speech model to serve it.
		for path, want := range map[string]int{"/v1/files": http.StatusOK, "/v1/audio/transcriptions": http.StatusBadRequest} {
			if got := post(path); got != want {
				t.Errorf("%s %s: status %d, want %d", name, path, got, want)
			}
			if left := removeTempFiles(t, dir); len(left) != 0 {
				t.Errorf("%s %s left temp files: %v", name, path, left)
			}
		}
	}
}

// A chunked body has no Content-Length for maxBytes' pre-check, so the read is what hits the cap. These handlers read
// the body themselves, and answer that 413, naming the limit, as decodeJSON does.
func TestUploads_chunkedBodyOverTheCapIs413(t *testing.T) {
	srv, err := newServer(config{})
	if err != nil {
		t.Fatal(err)
	}
	mux, _ := srv.routes(config{maxBodyBytes: 1 << 20}, "", nil)
	form, ctype := uploadBody(t, make([]byte, 2<<20))
	for _, c := range []struct {
		path, ctype string
		body        []byte
	}{
		{"/v1/audio/transcriptions", ctype, form},
		{"/v1/files", ctype, form},
		{"/v1/systemone", "application/json", []byte(`{"state":"` + strings.Repeat("a", 2<<20) + `"}`)},
	} {
		req := httptest.NewRequest(http.MethodPost, c.path, io.MultiReader(bytes.NewReader(c.body)))
		if req.ContentLength != -1 {
			t.Fatalf("%s: request length %d, want unknown: the pre-check would answer, not the read", c.path, req.ContentLength)
		}
		req.Header.Set("Content-Type", c.ctype)
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != http.StatusRequestEntityTooLarge || !strings.Contains(rec.Body.String(), "1048576-byte limit") {
			t.Errorf("%s: %d %q, want 413 naming the limit", c.path, rec.Code, rec.Body.String())
		}
	}
}

// bytesAllocatedBy is the heap f allocates, freed or not.
func bytesAllocatedBy(f func()) uint64 {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// A clip over its limit is refused from its header. Decoding this one would allocate 80 MiB, ten times its size:
// 16 MiB of samples and, because 4 kHz resamples up by 4, 64 MiB more.
func TestAudio_longClipRefusedBeforeDecoding(t *testing.T) {
	wav := wavBytes(4<<20, 4000) // 8 MiB of PCM, 1,048 s

	t.Run("transcriptions", func(t *testing.T) {
		s, st := speechServer(t)
		body, ctype := uploadBody(t, wav)
		upload := func() *http.Request {
			req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", bytes.NewReader(body))
			req.Header.Set("Content-Type", ctype)
			return req
		}
		// Holding the upload, which no refusal avoids: multipart's buffer and the bytes read from it.
		holding := bytesAllocatedBy(func() {
			req := upload()
			if err := req.ParseMultipartForm(maxFileUploadMemory); err != nil {
				t.Fatal(err)
			}
			fh := req.MultipartForm.File["file"][0]
			f, err := fh.Open()
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if _, err := io.ReadFull(f, make([]byte, fh.Size)); err != nil {
				t.Fatal(err)
			}
		})
		req, rec := upload(), httptest.NewRecorder()
		got := bytesAllocatedBy(func() { s.handleTranscriptions(rec, req) })
		if want := fmt.Sprintf("over the %d s", speechMaxSeconds); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), want) {
			t.Errorf("%d %q, want 400 containing %q", rec.Code, rec.Body.String(), want)
		}
		if st.n != 0 {
			t.Errorf("the engine saw %d samples of a refused clip", st.n)
		}
		if limit := holding + uint64(len(wav)); got > limit { // decoding would add ten times len(wav)
			t.Errorf("a refused clip allocated %d bytes, holding its upload takes %d, want under %d", got, holding, limit)
		}
	})

	lm := &loadedModel{}
	for name, c := range map[string]struct {
		prompt func() (visionInput, error)
		limit  int
	}{
		"gemma4":    {func() (visionInput, error) { return lm.gemma4AudioPrompt(nil, "", nil, 0, imageRef{data: wav}) }, gemma4AudioMaxSeconds},
		"qwen3-asr": {func() (visionInput, error) { return lm.qwenASRPrompt("", imageRef{data: wav}) }, qwenASRMaxSeconds},
		"voxtral":   {func() (visionInput, error) { return lm.voxtralPrompt(imageRef{data: wav}) }, voxtralMaxSeconds},
	} {
		t.Run(name, func(t *testing.T) {
			var err error
			got := bytesAllocatedBy(func() { _, err = c.prompt() })
			if want := fmt.Sprintf("over the %d s", c.limit); err == nil || !strings.Contains(err.Error(), want) {
				t.Errorf("error %v, want one containing %q", err, want)
			}
			if got > uint64(len(wav)) { // its samples alone are twice this
				t.Errorf("a refused clip allocated %d bytes, want none of its samples", got)
			}
		})
	}
}

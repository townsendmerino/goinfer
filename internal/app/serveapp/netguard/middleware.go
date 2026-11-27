package netguard

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"reflect"
	"strings"
	"time"
)

// Default request-body size ceilings and HTTP status constants.
const (
	MaxBodyBytes        = 4 << 20
	MaxVisionBodyBytes  = 32 << 20
	MaxEmbedBodyBytes   = 64 << 20
	MaxBatchFileBytes   = 64 << 20
	MaxFileUploadMemory = 32 << 20
	StatusCancelled     = 499
)

// JSONWriteTimeout bounds non-streaming JSON response writes.
var JSONWriteTimeout = 30 * time.Second

// WriteJSON encodes v to a buffer before writing code and application/json headers,
// returning a 500 api_error if JSON encoding fails.
func WriteJSON(w http.ResponseWriter, code int, v any) {
	_ = http.NewResponseController(w).SetWriteDeadline(time.Now().Add(JSONWriteTimeout))
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	if err := enc.Encode(v); err != nil {
		msg, _ := json.Marshal("response could not be encoded: " + err.Error())
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, `{"error":{"message":%s,"type":"api_error"}}`+"\n", msg)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_, _ = w.Write(buf.Bytes())
}

// WriteErr writes an OpenAI-shaped JSON error with type derived from code.
func WriteErr(w http.ResponseWriter, code int, msg string) {
	typ := "invalid_request_error"
	switch {
	case code == http.StatusTooManyRequests:
		typ = "rate_limit_error"
	case code >= 500:
		typ = "api_error"
	}
	WriteJSON(w, code, map[string]any{"error": map[string]any{"message": msg, "type": typ}})
}

// WriteServerErr writes a 500 api_error JSON response.
func WriteServerErr(w http.ResponseWriter, msg string) {
	WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]any{"message": msg, "type": "api_error"}})
}

// MaxBytes bounds r.Body to n bytes via a Content-Length pre-check and http.MaxBytesReader.
func MaxBytes(n int64, h http.HandlerFunc, note ...string) http.HandlerFunc {
	extra := ""
	if len(note) > 0 && note[0] != "" {
		extra = " " + note[0]
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if n > 0 && r.ContentLength > n {
			WriteErr(w, http.StatusRequestEntityTooLarge,
				fmt.Sprintf("request body is %d bytes, which exceeds the %d-byte limit (raise it with -max-body-bytes)%s", r.ContentLength, n, extra))
			return
		}
		if n > 0 {
			r.Body = http.MaxBytesReader(w, r.Body, n)
		}
		h(w, r)
	}
}

// LimitInflight caps concurrent handler executions via sem, returning 503 + Retry-After when full.
func LimitInflight(sem chan struct{}, h http.HandlerFunc) http.HandlerFunc {
	if sem == nil {
		return h
	}
	return func(w http.ResponseWriter, r *http.Request) {
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
			h(w, r)
		default:
			w.Header().Set("Retry-After", "1")
			WriteErr(w, http.StatusServiceUnavailable, "server at capacity (max in-flight requests reached); retry")
		}
	}
}

// RequireAuth enforces a shared secret via x-api-key or Bearer Authorization using
// constant-time SHA-256 digest comparison when key is non-empty.
func RequireAuth(key string, h http.HandlerFunc) http.HandlerFunc {
	if key == "" {
		return h
	}
	want := sha256.Sum256([]byte(key))
	return func(w http.ResponseWriter, r *http.Request) {
		got := r.Header.Get("x-api-key")
		if got == "" {
			got = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		}
		if sum := sha256.Sum256([]byte(got)); subtle.ConstantTimeCompare(sum[:], want[:]) != 1 {
			WriteErr(w, http.StatusUnauthorized, "missing or invalid API key")
			return
		}
		h(w, r)
	}
}

// BodyTooLarge reports whether err is an *http.MaxBytesError and formats its 413 message.
func BodyTooLarge(err error) (msg string, tooLarge bool) {
	if mbe, ok := errors.AsType[*http.MaxBytesError](err); ok {
		return fmt.Sprintf("request body exceeds the %d-byte limit (raise it with -max-body-bytes)", mbe.Limit), true
	}
	return "", false
}

// WriteBodyErr responds 413 when err exceeded MaxBytes, else 400 with msg and err.
func WriteBodyErr(w http.ResponseWriter, msg string, err error) {
	if tooLarge, ok := BodyTooLarge(err); ok {
		WriteErr(w, http.StatusRequestEntityTooLarge, tooLarge)
		return
	}
	WriteErr(w, http.StatusBadRequest, msg+": "+err.Error())
}

// ParseUpload parses r's multipart/form-data body bounded by MaxFileUploadMemory.
func ParseUpload(w http.ResponseWriter, r *http.Request, msg string) (form *multipart.Form, ok bool) {
	if err := r.ParseMultipartForm(MaxFileUploadMemory); err != nil {
		WriteBodyErr(w, msg, err)
		return nil, false
	}
	return r.MultipartForm, true
}

// JSONDecodeMessage shapes a JSON decode error without leaking internal Go struct or type names.
func JSONDecodeMessage(err error) (msg string, tooLarge bool) {
	if msg, tooLarge := BodyTooLarge(err); tooLarge {
		return msg, true
	}
	if ute, ok := errors.AsType[*json.UnmarshalTypeError](err); ok {
		switch ute.Type.Kind() {
		case reflect.Bool, reflect.String,
			reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
			reflect.Float32, reflect.Float64:
			return fmt.Sprintf("invalid request body: field %q has the wrong type (expected %s)", ute.Field, ute.Type.Kind()), false
		default:
			return fmt.Sprintf("invalid request body: field %q has the wrong type", ute.Field), false
		}
	}
	if se, ok := errors.AsType[*json.SyntaxError](err); ok {
		return fmt.Sprintf("invalid request body: malformed JSON at byte %d", se.Offset), false
	}
	return "invalid request body", false
}

// DecodeJSON decodes r.Body into v, writing a 413 or 400 JSON error and returning false on failure.
func DecodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		msg, tooLarge := JSONDecodeMessage(err)
		if tooLarge {
			WriteErr(w, http.StatusRequestEntityTooLarge, msg)
			return false
		}
		WriteErr(w, http.StatusBadRequest, msg)
		return false
	}
	return true
}

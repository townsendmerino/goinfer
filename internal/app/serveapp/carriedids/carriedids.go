// Package carriedids signs, parses, verifies, and re-aligns carried token ids on /v1/responses and /v1/messages.
//
// A reply's generated token ids ride back to the server in the reasoning item's encrypted_content, signed with a key only
// the server holds, so an echoed reply re-tokenizes to the ids the model generated rather than to the tokenizer's own split
// of their text. The ids are used for a turn only when the blob verifies and they decode exactly to the turn's text as the
// template renders it (or differ only in template-owned formatting); any other outcome tokenizes the turn as a request with
// no blob would, and is counted by reason.
//
// Design and gates: docs/server.md ("Carried token ids on /v1/responses") and
// docs/measurements/integration-harness-2026-10/id-roundtrip-step0-2026-10-15.md, §(a) to §(c).
package carriedids

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
)

// V2Prefix marks an encrypted_content that carries the reply's ids beside its reasoning text.
const V2Prefix = "goinfer-reasoning-v2:"

// BlobVersion is the first byte of a v2 payload.
const BlobVersion = 2

// The reasons a carried blob was not used, as -log-requests and /admin/status name them.
const (
	ReasonV1      = "v1"      // the blob carries no ids
	ReasonParse   = "parse"   // a malformed blob
	ReasonMAC     = "mac"     // a bad or missing signature
	ReasonKey     = "key"     // signed by another key (a restart with the default key, or another server)
	ReasonModel   = "model"   // minted for another model or tokenizer
	ReasonNS      = "ns"      // minted in another namespace
	ReasonSpecial = "special" // the special ids differ from the template's for the turn, or one sits inside content
	ReasonDecode  = "decode"  // the ids do not decode to the turn's text, and the difference is content: a client edit
	ReasonHistory = "history" // the template does not render this turn's reasoning (its history rule), so the ids cannot match it
	// The ids differ from the turn's render only in template-owned formatting (Format) but cannot be placed: no single run
	// of the render covers the turn and lines up with its special tokens.
	ReasonTemplate = "template"
)

// Reasons is every reason, in the order /admin/status lists them.
var Reasons = []string{
	ReasonV1, ReasonParse, ReasonMAC, ReasonKey, ReasonModel, ReasonNS,
	ReasonSpecial, ReasonDecode, ReasonHistory, ReasonTemplate,
}

// Key is the server's signing key and its id, the first 8 bytes of SHA-256 of the key.
type Key struct {
	key []byte
	ID  [8]byte
}

// NewKey wraps raw key bytes and computes their 8-byte SHA-256 key id.
func NewKey(key []byte) *Key {
	k := &Key{key: slices.Clone(key)}
	sum := sha256.Sum256(key)
	copy(k.ID[:], sum[:8])
	return k
}

// RandomKey is the default key: 32 bytes from crypto/rand, made at start and never written anywhere. A restart, or
// another server, makes older blobs fall back as `key`.
func RandomKey() (*Key, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return nil, fmt.Errorf("carried ids: no key: %w", err)
	}
	return NewKey(b), nil
}

// LoadKey is the key --ids-key-file names, or a random one when path is "".
func LoadKey(path string) (*Key, error) {
	if path == "" {
		return RandomKey()
	}
	return ReadKeyFile(path)
}

// KeyFileWarning is the startup warning for --ids-key-file on an OS where ReadKeyFile cannot check who may read the
// file, "" elsewhere or when no file is named.
func KeyFileWarning(goos, path string) string {
	if goos != "windows" || path == "" {
		return ""
	}
	return "warning: --ids-key-file " + path + ": permissions not checked on Windows; make sure only this account can read the key"
}

// ReadKeyFile is --ids-key-file: the key is the file's bytes, at least 32 of them, in a file no one but its owner can read.
// The mode is not checked on Windows, whose mode bits do not carry it (every file reads as 0666).
func ReadKeyFile(path string) (*Key, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("--ids-key-file: %w", err)
	}
	if runtime.GOOS != "windows" && fi.Mode().Perm()&0o077 != 0 {
		return nil, fmt.Errorf("--ids-key-file %s: mode %v lets others read the key; chmod 600 it", path, fi.Mode().Perm())
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("--ids-key-file: %w", err)
	}
	if len(b) < 32 {
		return nil, fmt.Errorf("--ids-key-file %s: %d bytes; the key needs at least 32", path, len(b))
	}
	return NewKey(b), nil
}

// Blob is a v2 encrypted_content. Whole says the ids cover the whole assistant turn; a streamed reply whose reasoning item
// closed before the turn ended carries the ids up to the reasoning's end instead. The MAC covers every field but Text: the
// text is checked against the ids instead, so an edit to it is a decode fallback, not a forgery.
type Blob struct {
	Whole bool
	KeyID [8]byte
	Model string
	NS    string
	Text  string
	IDs   []int
	MAC   []byte // nil when the blob carries none
}

func appendIDsString(b []byte, s string) []byte {
	b = binary.AppendUvarint(b, uint64(len(s)))
	return append(b, s...)
}

// signed is the bytes the MAC covers: version, flags, key id, model, namespace and the packed ids.
func (b *Blob) signed() []byte {
	out := []byte{BlobVersion, b.flags()}
	out = append(out, b.KeyID[:]...)
	out = appendIDsString(out, b.Model)
	out = appendIDsString(out, b.NS)
	return appendPackedIDs(out, b.IDs)
}

func (b *Blob) flags() byte {
	if b.Whole {
		return 1
	}
	return 0
}

func appendPackedIDs(out []byte, ids []int) []byte {
	out = binary.AppendUvarint(out, uint64(len(ids)))
	for _, id := range ids {
		out = binary.AppendUvarint(out, uint64(id))
	}
	return out
}

// String is the blob as encrypted_content: version, flags, key id, model, namespace, text, ids, then the MAC (length-prefixed,
// so a blob with none is still well formed), base64 after the prefix.
func (b *Blob) String() string {
	out := []byte{BlobVersion, b.flags()}
	out = append(out, b.KeyID[:]...)
	out = appendIDsString(out, b.Model)
	out = appendIDsString(out, b.NS)
	out = appendIDsString(out, b.Text)
	out = appendPackedIDs(out, b.IDs)
	out = binary.AppendUvarint(out, uint64(len(b.MAC)))
	out = append(out, b.MAC...)
	return V2Prefix + base64.StdEncoding.EncodeToString(out)
}

// ErrBlob is returned by ParseBlob for any malformed v2 encrypted_content.
var ErrBlob = errors.New("not a well-formed v2 blob")

// ParseBlob reads a v2 encrypted_content, prefix included. Any short, long or out-of-range field is ErrBlob.
func ParseBlob(enc string) (*Blob, error) {
	b64, ok := strings.CutPrefix(enc, V2Prefix)
	if !ok {
		return nil, ErrBlob
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil || len(raw) < 10 || raw[0] != BlobVersion || raw[1] > 1 {
		return nil, ErrBlob
	}
	b := &Blob{Whole: raw[1] == 1}
	copy(b.KeyID[:], raw[2:10])
	r := raw[10:]
	str := func() (string, bool) {
		n, k := binary.Uvarint(r)
		if k <= 0 || n > uint64(len(r)-k) {
			return "", false
		}
		s := string(r[k : k+int(n)])
		r = r[k+int(n):]
		return s, true
	}
	var okm, okn, okt bool
	if b.Model, okm = str(); !okm {
		return nil, ErrBlob
	}
	if b.NS, okn = str(); !okn {
		return nil, ErrBlob
	}
	if b.Text, okt = str(); !okt {
		return nil, ErrBlob
	}
	n, k := binary.Uvarint(r)
	if k <= 0 || n > uint64(len(r)) {
		return nil, ErrBlob
	}
	r = r[k:]
	b.IDs = make([]int, 0, n)
	for range n {
		id, k := binary.Uvarint(r)
		if k <= 0 || id > 1<<31-1 {
			return nil, ErrBlob
		}
		b.IDs = append(b.IDs, int(id))
		r = r[k:]
	}
	m, k := binary.Uvarint(r)
	if k <= 0 || m != uint64(len(r)-k) {
		return nil, ErrBlob
	}
	if m > 0 {
		b.MAC = slices.Clone(r[k:])
	}
	return b, nil
}

// Seal stamps the blob with this key's id and MAC.
func (k *Key) Seal(b *Blob) {
	b.KeyID = k.ID
	m := hmac.New(sha256.New, k.key)
	m.Write(b.signed())
	b.MAC = m.Sum(nil)
}

// VerifyMAC reports whether b's MAC is this key's over b's signed fields.
func (k *Key) VerifyMAC(b *Blob) bool {
	if len(b.MAC) != sha256.Size {
		return false
	}
	m := hmac.New(sha256.New, k.key)
	m.Write(b.signed())
	return hmac.Equal(m.Sum(nil), b.MAC)
}

// Mint is the encrypted_content of a reply's reasoning item carrying ids, signed by k.
func (k *Key) Mint(model, ns, text string, ids []int, whole bool) string {
	b := &Blob{Whole: whole, Model: model, NS: ns, Text: text, IDs: slices.Clone(ids)}
	k.Seal(b)
	return b.String()
}

// Carrier is what a reasoning input item carried for the assistant turn it attaches to: a parsed blob, or the reason it
// cannot be used (v1, parse) known already at parse time.
type Carrier struct {
	Blob   *Blob
	Reason string
}

// Outcome is one request's carried-id record: the assistant turns that carried a blob, those that used its ids, and the
// reason for each that did not, in turn order.
type Outcome struct {
	Carried, Used int
	Formatted     int // of Used, the turns whose ids replaced template-owned formatting (Format)
	Fallbacks     []string
}

// Note records one carried turn: reason == "" counts as used; otherwise it is appended to Fallbacks.
func (o *Outcome) Note(reason string) {
	o.Carried++
	if reason == "" {
		o.Used++
		return
	}
	o.Fallbacks = append(o.Fallbacks, reason)
}

// Fields is the outcome as -log-requests writes it: ids=<used>/<carried> ids_fallback=<reasons, comma-separated, or ->.
func (o *Outcome) Fields() string {
	fb := "-"
	if len(o.Fallbacks) > 0 {
		fb = strings.Join(o.Fallbacks, ",")
	}
	s := fmt.Sprintf("ids=%d/%d ids_fallback=%s", o.Used, o.Carried, fb)
	if o.Formatted > 0 {
		s += fmt.Sprintf(" ids_fmt=%d", o.Formatted)
	}
	return s
}

// Counters are the server's totals since start, for /admin/status.
type Counters struct {
	mu                       sync.Mutex
	carried, used, formatted int
	by                       map[string]int
}

// Add folds one request's outcome into the running totals.
func (c *Counters) Add(o Outcome) {
	if o.Carried == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.carried += o.Carried
	c.used += o.Used
	c.formatted += o.Formatted
	if c.by == nil {
		c.by = map[string]int{}
	}
	for _, r := range o.Fallbacks {
		c.by[r]++
	}
}

// Report snapshots the totals in the map layout /admin/status exposes.
func (c *Counters) Report() map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	fb := map[string]int{}
	for _, r := range Reasons {
		if n := c.by[r]; n > 0 {
			fb[r] = n
		}
	}
	return map[string]any{"carried": c.carried, "used": c.used, "formatted": c.formatted, "fallbacks": fb}
}

// Bytes is ids as the token-id log hashes them: each id a little-endian uint32.
func Bytes(ids []int) []byte {
	out := make([]byte, 0, 4*len(ids))
	for _, id := range ids {
		out = binary.LittleEndian.AppendUint32(out, uint32(id))
	}
	return out
}

// SHA256 is the hex SHA-256 of Bytes(ids).
func SHA256(ids []int) string {
	s := sha256.Sum256(Bytes(ids))
	return hex.EncodeToString(s[:])
}

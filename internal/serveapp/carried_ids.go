package serveapp

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
	"strconv"
	"strings"
	"sync"
)

// Carried ids (docs/server.md, "Carried token ids on /v1/responses"; design and gates: docs/measurements/integration-harness-2026-10/
// id-roundtrip-step0-2026-10-15.md, §(a) to §(c)). A reply's generated token ids ride back to the server in the reasoning
// item's encrypted_content, signed with a key only the server holds, so an echoed reply re-tokenizes to the ids the model
// generated rather than to the tokenizer's own split of their text. The ids are used for a turn only when the blob verifies
// and they decode exactly to the turn's text as the template renders it; any other outcome tokenizes the turn as a request
// with no blob would, and is counted by reason.

// reasoningBlobV2Prefix marks an encrypted_content that carries the reply's ids beside its reasoning text.
const reasoningBlobV2Prefix = "goinfer-reasoning-v2:"

// idsBlobVersion is the first byte of a v2 payload.
const idsBlobVersion = 2

// The reasons a carried blob was not used, as -log-requests and /admin/status name them.
const (
	idsV1      = "v1"      // the blob carries no ids
	idsParse   = "parse"   // a malformed blob
	idsMAC     = "mac"     // a bad or missing signature
	idsKeyR    = "key"     // signed by another key (a restart with the default key, or another server)
	idsModel   = "model"   // minted for another model or tokenizer
	idsNS      = "ns"      // minted in another namespace
	idsSpecial = "special" // the special ids differ from the template's for the turn, or one sits inside content
	idsDecode  = "decode"  // the ids do not decode to the turn's text, and the difference is content: a client edit
	idsHistory = "history" // the template does not render this turn's reasoning (its history rule), so the ids cannot match it
	// The ids differ from the turn's render only in template-owned formatting (idsFormat) but cannot be placed: no single run
	// of the render covers the turn and lines up with its special tokens.
	idsTemplate = "template"
)

// idsReasons is every reason, in the order /admin/status lists them.
var idsReasons = []string{idsV1, idsParse, idsMAC, idsKeyR, idsModel, idsNS, idsSpecial, idsDecode, idsHistory, idsTemplate}

// idsKey is the server's signing key and its id, the first 8 bytes of SHA-256 of the key.
type idsKey struct {
	key []byte
	id  [8]byte
}

func newIDsKey(key []byte) *idsKey {
	k := &idsKey{key: append([]byte(nil), key...)}
	sum := sha256.Sum256(key)
	copy(k.id[:], sum[:8])
	return k
}

// randomIDsKey is the default key: 32 bytes from crypto/rand, made at start and never written anywhere. A restart, or
// another server, makes older blobs fall back as `key`.
func randomIDsKey() (*idsKey, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return nil, fmt.Errorf("carried ids: no key: %w", err)
	}
	return newIDsKey(b), nil
}

// invertedBool is a boolean flag stored negated, so a flag that defaults to true leaves its config field at the zero value:
// --carry-ids sets config.noCarryIDs to its opposite.
type invertedBool struct{ p *bool }

func (v invertedBool) String() string {
	if v.p == nil {
		return "true"
	}
	return strconv.FormatBool(!*v.p)
}

func (v invertedBool) Set(s string) error {
	b, err := strconv.ParseBool(s)
	if err != nil {
		return err
	}
	*v.p = !b
	return nil
}

func (v invertedBool) IsBoolFlag() bool { return true }

// loadIDsKey is the key --ids-key-file names, or a random one when path is "".
func loadIDsKey(path string) (*idsKey, error) {
	if path == "" {
		return randomIDsKey()
	}
	return readIDsKeyFile(path)
}

// idsKeyFileWarning is the startup warning for --ids-key-file on an OS where readIDsKeyFile cannot check who may read the
// file, "" elsewhere or when no file is named.
func idsKeyFileWarning(goos, path string) string {
	if goos != "windows" || path == "" {
		return ""
	}
	return "warning: --ids-key-file " + path + ": permissions not checked on Windows; make sure only this account can read the key"
}

// readIDsKeyFile is --ids-key-file: the key is the file's bytes, at least 32 of them, in a file no one but its owner can read.
// The mode is not checked on Windows, whose mode bits do not carry it (every file reads as 0666).
func readIDsKeyFile(path string) (*idsKey, error) {
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
	return newIDsKey(b), nil
}

// idsBlob is a v2 encrypted_content. whole says the ids cover the whole assistant turn; a streamed reply whose reasoning item
// closed before the turn ended carries the ids up to the reasoning's end instead. The MAC covers every field but text: the text
// is checked against the ids instead, so an edit to it is a decode fallback, not a forgery.
type idsBlob struct {
	whole bool
	keyID [8]byte
	model string
	ns    string
	text  string
	ids   []int
	mac   []byte // nil when the blob carries none
}

func appendIDsString(b []byte, s string) []byte {
	b = binary.AppendUvarint(b, uint64(len(s)))
	return append(b, s...)
}

// signed is the bytes the MAC covers: version, flags, key id, model, namespace and the packed ids.
func (b *idsBlob) signed() []byte {
	out := []byte{idsBlobVersion, b.flags()}
	out = append(out, b.keyID[:]...)
	out = appendIDsString(out, b.model)
	out = appendIDsString(out, b.ns)
	return appendPackedIDs(out, b.ids)
}

func (b *idsBlob) flags() byte {
	if b.whole {
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
func (b *idsBlob) String() string {
	out := []byte{idsBlobVersion, b.flags()}
	out = append(out, b.keyID[:]...)
	out = appendIDsString(out, b.model)
	out = appendIDsString(out, b.ns)
	out = appendIDsString(out, b.text)
	out = appendPackedIDs(out, b.ids)
	out = binary.AppendUvarint(out, uint64(len(b.mac)))
	out = append(out, b.mac...)
	return reasoningBlobV2Prefix + base64.StdEncoding.EncodeToString(out)
}

var errIDsBlob = errors.New("not a well-formed v2 blob")

// parseIDsBlob reads a v2 encrypted_content, prefix included. Any short, long or out-of-range field is errIDsBlob.
func parseIDsBlob(enc string) (*idsBlob, error) {
	b64, ok := strings.CutPrefix(enc, reasoningBlobV2Prefix)
	if !ok {
		return nil, errIDsBlob
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil || len(raw) < 10 || raw[0] != idsBlobVersion || raw[1] > 1 {
		return nil, errIDsBlob
	}
	b := &idsBlob{whole: raw[1] == 1}
	copy(b.keyID[:], raw[2:10])
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
	if b.model, okm = str(); !okm {
		return nil, errIDsBlob
	}
	if b.ns, okn = str(); !okn {
		return nil, errIDsBlob
	}
	if b.text, okt = str(); !okt {
		return nil, errIDsBlob
	}
	n, k := binary.Uvarint(r)
	if k <= 0 || n > uint64(len(r)) {
		return nil, errIDsBlob
	}
	r = r[k:]
	b.ids = make([]int, 0, n)
	for range n {
		id, k := binary.Uvarint(r)
		if k <= 0 || id > 1<<31-1 {
			return nil, errIDsBlob
		}
		b.ids = append(b.ids, int(id))
		r = r[k:]
	}
	m, k := binary.Uvarint(r)
	if k <= 0 || m != uint64(len(r)-k) {
		return nil, errIDsBlob
	}
	if m > 0 {
		b.mac = append([]byte(nil), r[k:]...)
	}
	return b, nil
}

// seal stamps the blob with this key's id and MAC.
func (k *idsKey) seal(b *idsBlob) {
	b.keyID = k.id
	m := hmac.New(sha256.New, k.key)
	m.Write(b.signed())
	b.mac = m.Sum(nil)
}

// verifyMAC reports whether b's MAC is this key's over b's signed fields.
func (k *idsKey) verifyMAC(b *idsBlob) bool {
	if len(b.mac) != sha256.Size {
		return false
	}
	m := hmac.New(sha256.New, k.key)
	m.Write(b.signed())
	return hmac.Equal(m.Sum(nil), b.mac)
}

// mint is the encrypted_content of a reply's reasoning item carrying ids, signed by k.
func (k *idsKey) mint(model, ns, text string, ids []int, whole bool) string {
	b := &idsBlob{whole: whole, model: model, ns: ns, text: text, ids: append([]int(nil), ids...)}
	k.seal(b)
	return b.String()
}

// idsCarrier is what a reasoning input item carried for the assistant turn it attaches to: a parsed blob, or the reason it
// cannot be used (v1, parse) known already at parse time.
type idsCarrier struct {
	blob   *idsBlob
	reason string
}

// idsOutcome is one request's carried-id record: the assistant turns that carried a blob, those that used its ids, and the
// reason for each that did not, in turn order.
type idsOutcome struct {
	carried, used int
	formatted     int // of used, the turns whose ids replaced template-owned formatting (idsFormat)
	fallbacks     []string
}

func (o *idsOutcome) note(reason string) {
	o.carried++
	if reason == "" {
		o.used++
		return
	}
	o.fallbacks = append(o.fallbacks, reason)
}

// fields is the outcome as -log-requests writes it: ids=<used>/<carried> ids_fallback=<reasons, comma-separated, or ->.
func (o idsOutcome) fields() string {
	fb := "-"
	if len(o.fallbacks) > 0 {
		fb = strings.Join(o.fallbacks, ",")
	}
	s := fmt.Sprintf("ids=%d/%d ids_fallback=%s", o.used, o.carried, fb)
	if o.formatted > 0 {
		s += fmt.Sprintf(" ids_fmt=%d", o.formatted)
	}
	return s
}

// idsCounters are the server's totals since start, for /admin/status.
type idsCounters struct {
	mu                       sync.Mutex
	carried, used, formatted int
	by                       map[string]int
}

func (c *idsCounters) add(o idsOutcome) {
	if o.carried == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.carried += o.carried
	c.used += o.used
	c.formatted += o.formatted
	if c.by == nil {
		c.by = map[string]int{}
	}
	for _, r := range o.fallbacks {
		c.by[r]++
	}
}

func (c *idsCounters) report() map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	fb := map[string]int{}
	for _, r := range idsReasons {
		if n := c.by[r]; n > 0 {
			fb[r] = n
		}
	}
	return map[string]any{"carried": c.carried, "used": c.used, "formatted": c.formatted, "fallbacks": fb}
}

// idsBytes is ids as the token-id log hashes them: each id a little-endian uint32.
func idsBytes(ids []int) []byte {
	out := make([]byte, 0, 4*len(ids))
	for _, id := range ids {
		out = binary.LittleEndian.AppendUint32(out, uint32(id))
	}
	return out
}

// idsSHA256 is the hex SHA-256 of idsBytes(ids).
func idsSHA256(ids []int) string {
	s := sha256.Sum256(idsBytes(ids))
	return hex.EncodeToString(s[:])
}

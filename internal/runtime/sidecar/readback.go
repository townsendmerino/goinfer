package sidecar

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"strings"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/runtime/giw"
)

type sidecarHooksKey struct{}

// TestHooks holds optional test seams attached to a transcode context.
type TestHooks struct {
	AfterWrite func(tmp string)
	WrapSink   func(io.Writer) io.Writer
}

func WithTestHooks(ctx context.Context, h TestHooks) context.Context {
	return context.WithValue(ctx, sidecarHooksKey{}, h)
}

func TestHooksOf(ctx context.Context) TestHooks {
	if ctx == nil {
		return TestHooks{}
	}
	h, _ := ctx.Value(sidecarHooksKey{}).(TestHooks)
	return h
}

// ErrReadBack is the read-back's verdict that the file does not hold what its CRC says.
var ErrReadBack = errors.New("read-back: the weights read from the file do not match their CRC")

// ReadBack checks a weights blob while it is being written to f: a goroutine reads the file behind the writer and
// keeps a CRC-32 of what it reads, and finish compares it with the blob's trailing CRC word, as a load does. The bytes
// it reads are the system's copy of what was just written, usually still in the page cache, so it checks the file as
// the system holds it and not the disk (markWritten).
type ReadBack struct {
	f       *os.File
	base    int64         // the blob's offset in f
	written atomic.Int64  // blob bytes handed to f so far
	read    atomic.Int64  // blob bytes read back so far
	wake    chan struct{} // a write happened
	done    chan struct{} // closed when the last byte is written, or the write is abandoned
	res     chan error    // the goroutine's verdict
}

// StartReadBack begins reading behind a blob that starts at offset base of f.
func StartReadBack(f *os.File, base int64) *ReadBack {
	rb := &ReadBack{f: f, base: base, wake: make(chan struct{}, 1), done: make(chan struct{}), res: make(chan error, 1)}
	go func() { rb.res <- rb.run() }()
	return rb
}

// Writer returns the writer the blob is streamed through: w, with every write counted.
func (rb *ReadBack) Writer(w io.Writer) io.Writer { return readBackWriter{rb, w} }

type readBackWriter struct {
	rb *ReadBack
	w  io.Writer
}

func (w readBackWriter) Write(p []byte) (int, error) {
	n, err := w.w.Write(p)
	w.rb.written.Add(int64(n))
	select {
	case w.rb.wake <- struct{}{}:
	default:
	}
	return n, err
}

// run reads [0, written-4) of the blob as it grows; once done is closed it reads to the end and compares.
func (rb *ReadBack) run() error {
	const crcWord = 4
	buf := make([]byte, 1<<20)
	var off int64
	var crc uint32
	final := false
	for {
		limit := rb.written.Load() - crcWord
		for off < limit {
			n := min(int64(len(buf)), limit-off)
			if _, err := rb.f.ReadAt(buf[:n], rb.base+off); err != nil {
				return fmt.Errorf("read-back at %d: %w", rb.base+off, err)
			}
			crc = crc32.Update(crc, crc32.IEEETable, buf[:n])
			off += n
			rb.read.Store(off)
		}
		if final {
			if limit < 0 {
				return fmt.Errorf("read-back: a weights blob of %d bytes has no CRC", limit+crcWord)
			}
			var word [crcWord]byte
			if _, err := rb.f.ReadAt(word[:], rb.base+limit); err != nil {
				return fmt.Errorf("read-back at %d: %w", rb.base+limit, err)
			}
			if want := binary.LittleEndian.Uint32(word[:]); crc != want {
				return fmt.Errorf("%w (read %08x, the file says %08x)", ErrReadBack, crc, want)
			}
			return nil
		}
		select {
		case <-rb.wake:
		case <-rb.done:
			final = true
		}
	}
}

// Finish is called once the blob's last byte is written: it waits for the read to reach the end and returns the verdict.
func (rb *ReadBack) Finish() error {
	close(rb.done)
	return <-rb.res
}

// writeSidecarBundle streams a bundle into f (giw.WriteStream) with the weights read back behind the write. An error from
// body wins over the read-back's.
func writeSidecarBundle(ctx context.Context, f *os.File, tok []byte, body func(io.Writer) (int64, error)) error {
	hooks := TestHooksOf(ctx)
	return giw.WriteStream(f, tok, func(w io.Writer) (int64, error) {
		base, err := f.Seek(0, io.SeekCurrent)
		if err != nil {
			return 0, err
		}
		if hooks.WrapSink != nil {
			w = hooks.WrapSink(w)
		}
		rb := StartReadBack(f, base)
		n, err := body(rb.Writer(w))
		verr := rb.Finish()
		if err != nil {
			return n, err
		}
		return n, verr
	})
}

// MarkWritten records the verified-marker for a bundle the transcode has written and read back, when the bundle and the
// source it was built from are together no larger than the host's memory. Then the self-check's load and the first real
// load skip the whole-file CRC (MarkGIWWritten).
//
// The source counts because the transcode maps it: a bundle that fits by itself is still pushed out of the page cache
// by a source that fills memory, and what the read-back saw is then no longer what a load reads. Such a bundle gets no
// marker, so the self-check's load runs the CRC over the file and reads that part from the disk. An unknown memory
// size (zero) or an unreadable source is treated the same way. Available memory is not the measure: a system counts
// the cache the source occupies as available.
func MarkWritten(tmp, source string, hostRAM func() int64, markGIWWritten func(string)) {
	total := hostRAM()
	fi, err := os.Stat(tmp)
	if err != nil || total <= 0 {
		return
	}
	if src := SourceBytes(source); src >= 0 && fi.Size()+src <= total {
		markGIWWritten(tmp)
	}
}

// SourceBytes is the size of the weights a transcode of source maps: the file itself, or a directory's .safetensors
// files. It is -1 when source cannot be read.
func SourceBytes(source string) int64 {
	fi, err := os.Stat(source)
	if err != nil {
		return -1
	}
	if !fi.IsDir() {
		return fi.Size()
	}
	entries, err := os.ReadDir(source)
	if err != nil {
		return -1
	}
	var n int64
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".safetensors") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			return -1
		}
		n += info.Size()
	}
	return n
}

// ReadBytes returns the total bytes read back so far.
func (rb *ReadBack) ReadBytes() int64 { return rb.read.Load() }

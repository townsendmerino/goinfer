package prequant

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"strings"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// hostRAM is the installed-memory probe markWritten compares a bundle and its source with.
var hostRAM = decoder.HostRAMBytes

// wrapSinkForTest, when set, wraps the writer the weights are streamed into, beneath the read-back.
var wrapSinkForTest func(io.Writer) io.Writer

// errReadBack is the read-back's verdict that the file does not hold what its CRC says.
var errReadBack = errors.New("read-back: the weights read from the file do not match their CRC")

// readBack checks a weights blob while it is being written to f: a goroutine reads the file behind the writer and
// keeps a CRC-32 of what it reads, and finish compares it with the blob's trailing CRC word, as a load does. The bytes
// it reads are the system's copy of what was just written, usually still in the page cache, so it checks the file as
// the system holds it and not the disk (markWritten).
type readBack struct {
	f       *os.File
	base    int64         // the blob's offset in f
	written atomic.Int64  // blob bytes handed to f so far
	read    atomic.Int64  // blob bytes read back so far
	wake    chan struct{} // a write happened
	done    chan struct{} // closed when the last byte is written, or the write is abandoned
	res     chan error    // the goroutine's verdict
}

// startReadBack begins reading behind a blob that starts at offset base of f.
func startReadBack(f *os.File, base int64) *readBack {
	rb := &readBack{f: f, base: base, wake: make(chan struct{}, 1), done: make(chan struct{}), res: make(chan error, 1)}
	go func() { rb.res <- rb.run() }()
	return rb
}

// writer returns the writer the blob is streamed through: w, with every write counted.
func (rb *readBack) writer(w io.Writer) io.Writer { return readBackWriter{rb, w} }

type readBackWriter struct {
	rb *readBack
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
func (rb *readBack) run() error {
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
				return fmt.Errorf("%w (read %08x, the file says %08x)", errReadBack, crc, want)
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

// finish is called once the blob's last byte is written: it waits for the read to reach the end and returns the verdict.
func (rb *readBack) finish() error {
	close(rb.done)
	return <-rb.res
}

// writeBundle streams a bundle into f (giw.WriteStream) with the weights read back behind the write. An error from
// body wins over the read-back's.
func writeBundle(f *os.File, tok []byte, body func(io.Writer) (int64, error)) error {
	return giw.WriteStream(f, tok, func(w io.Writer) (int64, error) {
		base, err := f.Seek(0, io.SeekCurrent)
		if err != nil {
			return 0, err
		}
		if wrapSinkForTest != nil {
			w = wrapSinkForTest(w)
		}
		rb := startReadBack(f, base)
		n, err := body(rb.writer(w))
		verr := rb.finish()
		if err != nil {
			return n, err
		}
		return n, verr
	})
}

// markWritten records the verified-marker for a bundle the transcode has written and read back, when the bundle and the
// source it was built from are together no larger than the host's memory. Then the self-check's load and the first real
// load skip the whole-file CRC (decoder.MarkGIWWritten).
//
// The source counts because the transcode maps it: a bundle that fits by itself is still pushed out of the page cache
// by a source that fills memory, and what the read-back saw is then no longer what a load reads. Such a bundle gets no
// marker, so the self-check's load runs the CRC over the file and reads that part from the disk. An unknown memory
// size (zero) or an unreadable source is treated the same way. Available memory is not the measure: a system counts
// the cache the source occupies as available.
func markWritten(tmp, source string) {
	total := hostRAM()
	fi, err := os.Stat(tmp)
	if err != nil || total <= 0 {
		return
	}
	if src := sourceBytes(source); src >= 0 && fi.Size()+src <= total {
		decoder.MarkGIWWritten(tmp)
	}
}

// sourceBytes is the size of the weights a transcode of source maps: the file itself, or a directory's .safetensors
// files. It is -1 when source cannot be read.
func sourceBytes(source string) int64 {
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

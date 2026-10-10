package prequant

import (
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// flipLastPayloadByte flips the byte before a bundle's trailing CRC word and restores the file's mtime, so the file
// keeps the size and mtime its verified-marker names while no longer matching its own CRC.
func flipLastPayloadByte(t *testing.T, path string) {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, 1)
	off := fi.Size() - 5
	if _, err := f.ReadAt(b, off); err != nil {
		t.Fatal(err)
	}
	b[0] ^= 0xff
	if _, err := f.WriteAt(b, off); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, fi.ModTime(), fi.ModTime()); err != nil {
		t.Fatal(err)
	}
}

// TestTranscode_selfCheckRunsNoCRCPass: the transcode marks a bundle it wrote and read back, when the bundle fits in
// the available memory, so its self-check loads the bundle without a second read for the CRC, and the marker follows
// the bundle to its final name. The proof is a payload
// byte flipped between the write and the self-check: the transcode still succeeds and the sidecar still loads, and the
// same sidecar under GOINFER_GIW_VERIFY=always is refused, which shows the flip is one a CRC pass catches.
func TestTranscode_selfCheckRunsNoCRCPass(t *testing.T) {
	if v, was := os.LookupEnv("GOINFER_GIW_VERIFY"); was {
		os.Unsetenv("GOINFER_GIW_VERIFY")
		t.Cleanup(func() { os.Setenv("GOINFER_GIW_VERIFY", v) })
	}
	hostRAMAvailable = func() int64 { return 1 << 40 }
	t.Cleanup(func() { hostRAMAvailable = decoder.HostRAMAvailableBytes })
	hooked := 0
	afterWriteForTest = func(tmp string) {
		hooked++
		if _, err := os.Stat(decoder.GIWVerifiedMarkerPath(tmp)); err != nil {
			t.Errorf("the written bundle has no marker before its self-check: %v", err)
		}
		flipLastPayloadByte(t, tmp)
	}
	t.Cleanup(func() { afterWriteForTest = nil })

	src := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	out := filepath.Join(t.TempDir(), "p.giw")
	if err := Transcode(context.Background(), src, out, "int4", false, decoder.GIWTargetNone); err != nil {
		t.Fatalf("the self-check ran a CRC pass over a bundle the transcode had marked: %v", err)
	}
	if hooked != 1 {
		t.Fatalf("the hook ran %d times, want 1", hooked)
	}
	if _, err := os.Stat(decoder.GIWVerifiedMarkerPath(out)); err != nil {
		t.Fatalf("the marker did not follow the bundle to its final name: %v", err)
	}
	m, err := decoder.Load(out, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("the first load of the sidecar ran a CRC pass: %v", err)
	}
	m.Close()

	t.Setenv("GOINFER_GIW_VERIFY", "always")
	if _, err := decoder.Load(out, decoder.Options{Backend: "cpu"}); err == nil || !strings.Contains(err.Error(), "CRC") {
		t.Fatalf("GOINFER_GIW_VERIFY=always must run the CRC and refuse the flipped bundle, got: %v", err)
	}
}

// TestTranscode_selfCheckStillRefusesAnUnloadableBundle: the self-check still loads the bundle, so one that does not
// deserialize never becomes the sidecar. The bundle's header is overwritten between the write and the self-check, at
// the same size and mtime, so the marker still holds and the refusal is the load's own and not the CRC's.
func TestTranscode_selfCheckStillRefusesAnUnloadableBundle(t *testing.T) {
	afterWriteForTest = func(tmp string) {
		fi, err := os.Stat(tmp)
		if err != nil {
			t.Fatal(err)
		}
		f, err := os.OpenFile(tmp, os.O_RDWR, 0)
		if err != nil {
			t.Fatal(err)
		}
		junk := make([]byte, 64)
		for i := range junk {
			junk[i] = 0xff
		}
		if _, err := f.WriteAt(junk, 8); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(tmp, fi.ModTime(), fi.ModTime()); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { afterWriteForTest = nil })

	src := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	dir := t.TempDir()
	out := filepath.Join(dir, "p.giw")
	err := Transcode(context.Background(), src, out, "int4", false, decoder.GIWTargetNone)
	if err == nil {
		t.Fatal("a bundle with an overwritten header passed the self-check")
	}
	if strings.Contains(err.Error(), "CRC") {
		t.Fatalf("the refusal came from a CRC pass, not from the load: %v", err)
	}
	left, _ := os.ReadDir(dir)
	if len(left) != 0 {
		t.Fatalf("a failed transcode left %d files behind (%s first)", len(left), left[0].Name())
	}
}

// TestTranscode_largerThanRAMGetsTheCRCPass: a bundle larger than the memory that was available, or written on a host
// whose available memory is unknown, gets no marker from the transcode, so the self-check's load runs the CRC: a
// payload byte flipped after the write fails the transcode there. Left alone, the same bundle passes and carries the
// marker that load earned.
func TestTranscode_largerThanRAMGetsTheCRCPass(t *testing.T) {
	if v, was := os.LookupEnv("GOINFER_GIW_VERIFY"); was {
		os.Unsetenv("GOINFER_GIW_VERIFY")
		t.Cleanup(func() { os.Setenv("GOINFER_GIW_VERIFY", v) })
	}
	t.Cleanup(func() { hostRAMAvailable = decoder.HostRAMAvailableBytes; afterWriteForTest = nil })
	src := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	for _, avail := range []int64{1, 0} {
		hostRAMAvailable = func() int64 { return avail }
		for _, flip := range []bool{true, false} {
			afterWriteForTest = func(tmp string) {
				if _, err := os.Stat(decoder.GIWVerifiedMarkerPath(tmp)); err == nil {
					t.Errorf("available %d: the transcode marked a bundle larger than that", avail)
				}
				if flip {
					flipLastPayloadByte(t, tmp)
				}
			}
			dir := t.TempDir()
			out := filepath.Join(dir, "p.giw")
			err := Transcode(context.Background(), src, out, "int4", false, decoder.GIWTargetNone)
			if flip {
				if err == nil || !strings.Contains(err.Error(), "CRC") {
					t.Fatalf("available %d: a flipped byte must fail the self-check's CRC, got: %v", avail, err)
				}
				if left, _ := os.ReadDir(dir); len(left) != 0 {
					t.Fatalf("available %d: a failed transcode left %d files behind (%s first)", avail, len(left), left[0].Name())
				}
				continue
			}
			if err != nil {
				t.Fatalf("available %d: %v", avail, err)
			}
			if _, err := os.Stat(decoder.GIWVerifiedMarkerPath(out)); err != nil {
				t.Errorf("available %d: the self-check's CRC pass left no marker at the final name: %v", avail, err)
			}
		}
	}
}

// flipAt passes writes through with the byte at stream offset at inverted.
type flipAt struct {
	w       io.Writer
	at, off int64
}

func (f *flipAt) Write(p []byte) (int, error) {
	if i := f.at - f.off; i >= 0 && i < int64(len(p)) {
		q := append([]byte(nil), p...)
		q[i] ^= 0xff
		p = q
	}
	f.off += int64(len(p))
	return f.w.Write(p)
}

// TestTranscode_readBackCatchesAByteThatDiffersInTheFile: the weights are read back behind the write and compared
// with their own CRC, so a byte that reaches the file different from the one the writer summed fails the transcode
// before the self-check, and nothing is left behind. An untouched stream passes the same read-back.
func TestTranscode_readBackCatchesAByteThatDiffersInTheFile(t *testing.T) {
	t.Cleanup(func() { wrapSinkForTest = nil; afterWriteForTest = nil })
	src := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	reachedSelfCheck := false
	afterWriteForTest = func(string) { reachedSelfCheck = true }
	for _, at := range []int64{0, 4097, 100000} {
		wrapSinkForTest = func(w io.Writer) io.Writer { return &flipAt{w: w, at: at} }
		dir := t.TempDir()
		err := Transcode(context.Background(), src, filepath.Join(dir, "p.giw"), "int4", false, decoder.GIWTargetNone)
		if !errors.Is(err, errReadBack) {
			t.Fatalf("a byte flipped at %d of the weights: want the read-back's error, got: %v", at, err)
		}
		if reachedSelfCheck {
			t.Fatalf("a byte flipped at %d: the transcode went on to the self-check", at)
		}
		if left, _ := os.ReadDir(dir); len(left) != 0 {
			t.Fatalf("a failed transcode left %d files behind (%s first)", len(left), left[0].Name())
		}
	}
	wrapSinkForTest = func(w io.Writer) io.Writer { return &flipAt{w: w, at: -1} }
	if err := Transcode(context.Background(), src, filepath.Join(t.TempDir(), "p.giw"), "int4", false, decoder.GIWTargetNone); err != nil {
		t.Fatalf("an untouched stream: %v", err)
	}
}

// TestReadBack_readsBehindTheWriter: the read-back keeps up with the write instead of reading the file at the end.
// After every write but the last it is given time to catch up, and it must have read to within the CRC word of what
// was written.
func TestReadBack_readsBehindTheWriter(t *testing.T) {
	f, err := os.Create(filepath.Join(t.TempDir(), "blob"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	const base = 64
	if _, err := f.Write(make([]byte, base)); err != nil {
		t.Fatal(err)
	}
	rb := startReadBack(f, base)
	w := rb.writer(f)
	chunk := make([]byte, 300_000)
	var crc uint32
	for i := range 8 {
		for j := range chunk {
			chunk[j] = byte(i*31 + j)
		}
		crc = crc32.Update(crc, crc32.IEEETable, chunk)
		if _, err := w.Write(chunk); err != nil {
			t.Fatal(err)
		}
		want := int64(i+1)*int64(len(chunk)) - 4
		caught := false
		for range 2000 {
			if rb.read.Load() >= want {
				caught = true
				break
			}
			time.Sleep(time.Millisecond)
		}
		if !caught {
			t.Fatalf("after write %d the read-back had read %d bytes, want %d: it is not reading behind the writer", i, rb.read.Load(), want)
		}
	}
	if _, err := w.Write(binary.LittleEndian.AppendUint32(nil, crc)); err != nil {
		t.Fatal(err)
	}
	if err := rb.finish(); err != nil {
		t.Fatalf("a blob written whole with its own CRC: %v", err)
	}

	// A blob too short to hold a CRC word is an error, not a pass.
	short := startReadBack(f, base)
	if _, err := short.writer(io.Discard).Write([]byte{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := short.finish(); err == nil {
		t.Error("a two-byte blob passed the read-back")
	}
}

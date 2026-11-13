package decoder

import (
	"errors"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

var kvMissBackends atomic.Int64

// kvMissBackend's build fails the way the CUDA build's KV check does when a context does not fit what is free: an error, not
// a decline and not ErrLoadAborted.
type kvMissBackend struct{ Backend }

func (b *kvMissBackend) BuildResident(*Model) (ResidentForward, bool, error) {
	return nil, false, errors.New("cuda: resident context 12353 positions needs 0.40 GB of KV but only 0.39 GB is free")
}

// TestLoad_aBuildErrorContinuesOnTheCPU pins the path a resident build's own error takes: the load succeeds on the CPU with
// the error as its decline reason, rather than failing. The CUDA build's errKVWontFit takes it; only ErrLoadAborted fails the
// load.
func TestLoad_aBuildErrorContinuesOnTheCPU(t *testing.T) {
	name := "fake-kvmiss-" + strconv.FormatInt(kvMissBackends.Add(1), 10)
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		return &kvMissBackend{cpu}, err
	})
	m, err := Load(tinyFixture(t), Options{Backend: name})
	if err != nil {
		t.Fatalf("a resident build's error failed the load: %v", err)
	}
	defer m.Close()
	if m.resident != nil {
		t.Fatal("a failed build left a resident")
	}
	if got := m.ResidentDecline(); !strings.Contains(got, "needs 0.40 GB of KV") {
		t.Errorf("the decline reason %q does not carry the build's error", got)
	}
}

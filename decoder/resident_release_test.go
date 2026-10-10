package decoder

import "testing"

type releaseTestBackend struct {
	Backend
	decline bool
}

func (b *releaseTestBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	if b.decline {
		return nil, false, nil
	}
	_, _, _, _, _, _, vocab := m.Dims()
	return &erroringResident{vocab: vocab, okCount: 1 << 30}, true, nil
}

func (b *releaseTestBackend) Close() error { return nil }

// TestWithResidency_releasesHostMemoryOnlyAfterASuccessfulBuild pins the load-time release of host memory: Go's scavenger
// otherwise returns the host-side packing lazily, so a real gpt-oss-20b on CUDA sat at its peak RSS for minutes after the load
// finished. The release must happen after a resident build succeeds, and must NOT happen for a decline, which built nothing
// and would only pay for a pointless collection.
func TestWithResidency_releasesHostMemoryOnlyAfterASuccessfulBuild(t *testing.T) {
	calls := 0
	old := releaseHostMemory
	releaseHostMemory = func() { calls++ }
	t.Cleanup(func() { releaseHostMemory = old })

	for _, tc := range []struct {
		name    string
		decline bool
		want    int
	}{
		{"successful resident build", false, 1},
		{"declined resident build", true, 0},
	} {
		calls = 0
		be := &releaseTestBackend{decline: tc.decline}
		name := "release-test-" + tc.name
		RegisterBackend(name, func() (Backend, error) {
			cpu, err := NewBackend("cpu")
			if err != nil {
				return nil, err
			}
			be.Backend = cpu
			return be, nil
		})
		m, err := Load(tinyFixture(t), Options{Backend: name})
		if err != nil {
			t.Fatalf("%s: Load: %v", tc.name, err)
		}
		if !tc.decline && !m.ResidentActive() {
			m.Close()
			t.Skip("fixture is not resident-eligible")
		}
		if calls != tc.want {
			t.Errorf("%s: releaseHostMemory called %d time(s), want %d", tc.name, calls, tc.want)
		}
		_ = m.Close()
	}
}

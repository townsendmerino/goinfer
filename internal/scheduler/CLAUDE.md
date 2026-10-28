# internal/scheduler — working rules
- Day loop: `go test ./internal/scheduler/` and `go test ./internal/serveapp -run 'TestSchedCensus_admission|TestSchedCensus_placement'` (about 4 s). Run both for any change here.
- Import only the standard library (`internal/layout`'s `scheduler-standard-library` rule). Never take a `decoder.Session`, a clock or I/O; the caller computes the inputs.
- A scheduling change that moves the census is a behavior change: it needs Francis's word before `-sched.update` rewrites a golden.
- A new invariant goes in `doc.go` with its test name.

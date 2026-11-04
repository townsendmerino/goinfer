//go:build !goinfer_testhooks

package serveapp

// preamblePark is a no-op in production builds — an empty call the compiler inlines away, not a
// branch. Under -tags goinfer_testhooks it becomes a settable hook (preamblepark_testhooks.go) that
// the drain regression test uses to park a request inside the pick→enter window while holding the
// liveness RLock. It is called from withModel immediately after the RLock is taken; keeping it at the
// acquisition site is what keeps the test honest if pick/enter are ever reordered (see
// docs/completed/task-admin-unload-drain.md).
func preamblePark() {}

package serveapp

// fakeCloser stands in for a model in liveness and unload-drain tests without loading a checkpoint.
type fakeCloser struct{}

func (*fakeCloser) Close() error { return nil }

// fakeDrafter is an attached block drafter with no weights: it embeds a nil blockDrafter, so a generation through it
// panics; a test of whether one is attached needs nothing more.
type fakeDrafter struct{ blockDrafter }

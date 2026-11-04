package serveapp

// fakeModel is a model handle with no checkpoint, for tests of the serving shell. It embeds a nil modelHandle, so a
// method the test did not expect panics instead of answering; Close succeeds, as a zero *decoder.Model's does.
type fakeModel struct{ modelHandle }

func (*fakeModel) Close() error { return nil }

// fakeDrafter is an attached block drafter with no weights: it embeds a nil blockDrafter, so a generation through it
// panics; a test of whether one is attached needs nothing more.
type fakeDrafter struct{ blockDrafter }

package serveapp

// fakeModel is a model handle with no checkpoint, for tests of the serving shell. It embeds a nil modelHandle, so a
// method the test did not expect panics instead of answering; Close succeeds, as a zero *decoder.Model's does.
type fakeModel struct{ modelHandle }

func (*fakeModel) Close() error { return nil }

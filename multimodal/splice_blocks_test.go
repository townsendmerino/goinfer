package multimodal

import (
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// S11: several image blocks spliced from the end, each its own Special segment, the text between them left as text.
func TestSpliceImageBlocks(t *testing.T) {
	const a, b = "<A>", "<B>"
	segs := []tokenizer.Segment{{Text: "<sys>", Special: true}, {Text: "user\nfirst <A> mid <B> last"}, {Text: "<eot>", Special: true}}
	got, err := SpliceImageBlocks(segs, []string{a, b})
	if err != nil {
		t.Fatal(err)
	}
	want := []tokenizer.Segment{{Text: "<sys>", Special: true}, {Text: "user\nfirst "}, {Text: a, Special: true}, {Text: " mid "},
		{Text: b, Special: true}, {Text: " last"}, {Text: "<eot>", Special: true}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v\nwant %+v", got, want)
	}
	// The same block text twice (two same-size images), adjacent: two Special segments, in order.
	got, err = SpliceImageBlocks([]tokenizer.Segment{{Text: "q: <A><A>!"}}, []string{a, a})
	if err != nil {
		t.Fatal(err)
	}
	want = []tokenizer.Segment{{Text: "q: "}, {Text: a, Special: true}, {Text: a, Special: true}, {Text: "!"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("adjacent same blocks: got %+v\nwant %+v", got, want)
	}
	// An earlier turn quoting the block text is not taken for an image (V-19, per block).
	got, err = SpliceImageBlocks([]tokenizer.Segment{{Text: "old <A>"}, {Text: "<eot>", Special: true}, {Text: "new <A> and <A>"}}, []string{a, a})
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Special || got[0].Text != "old <A>" {
		t.Errorf("the earlier turn's text was spliced: %+v", got)
	}
	// One block missing: an error naming which image.
	if _, err := SpliceImageBlocks([]tokenizer.Segment{{Text: "only <A>"}}, []string{a, b}); err == nil {
		t.Error("a missing block spliced without error")
	}
}

func TestFindImageRuns(t *testing.T) {
	got := FindImageRuns([]int{1, 9, 9, 2, 3, 9, 9, 9, 4, 9}, 9)
	want := [][2]int{{1, 2}, {5, 3}, {9, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	if r := FindImageRuns([]int{1, 2}, 9); r != nil {
		t.Errorf("no runs: got %v", r)
	}
}

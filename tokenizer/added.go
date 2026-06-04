package tokenizer

// addedTrie matches Gemma's added/special tokens against raw text. HF's
// AddedVocabulary splits the input on these surface forms *before* the
// normalizer and BPE run; since all of Gemma's added tokens are
// non-normalized and neither lstrip nor rstrip, this reduces to a plain
// leftmost-longest substring match at each byte position.
//
// The trie is keyed by byte (not rune) so matching never has to decode UTF-8;
// a node carries id≥0 when some added token ends there. At a given position we
// walk as deep as the text allows and return the deepest node that was a token
// end — the longest match (so "\n\n" wins over "\n").
type addedTrie struct {
	root *trieNode
}

type trieNode struct {
	next map[byte]*trieNode
	id   int32 // ≥0 if a token ends here
	end  bool
}

func newAddedTrie() *addedTrie {
	return &addedTrie{root: &trieNode{next: map[byte]*trieNode{}}}
}

func (t *addedTrie) add(content string, id int32) {
	if content == "" {
		return
	}
	n := t.root
	for i := 0; i < len(content); i++ {
		c := content[i]
		child, ok := n.next[c]
		if !ok {
			child = &trieNode{next: map[byte]*trieNode{}}
			n.next[c] = child
		}
		n = child
	}
	n.end = true
	n.id = id
}

// match returns the id and byte length of the longest added token that the
// text matches starting at pos, or (0, 0) if none.
func (t *addedTrie) match(text string, pos int) (int32, int) {
	n := t.root
	bestID := int32(0)
	bestLen := 0
	for i := pos; i < len(text); i++ {
		child, ok := n.next[text[i]]
		if !ok {
			break
		}
		n = child
		if n.end {
			bestID = n.id
			bestLen = i - pos + 1
		}
	}
	return bestID, bestLen
}

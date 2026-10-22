package prefix

import (
	"strings"
	"testing"
)

func TestHashesChainPrefixes(t *testing.T) {
	ix := New(4, 100)
	a := ix.Hashes("aaaabbbbcccc")
	b := ix.Hashes("aaaabbbbdddd")
	c := ix.Hashes("xaaabbbbcccc")
	if len(a) != 3 || len(ix.Hashes("aaaab")) != 1 || ix.Hashes("abc") != nil {
		t.Fatalf("block counts: %d, partial blocks must be dropped", len(a))
	}
	if a[0] != b[0] || a[1] != b[1] || a[2] == b[2] {
		t.Error("prompts sharing two blocks should share exactly two hashes")
	}
	if a[1] == c[1] {
		t.Error("equal block content after a different prefix must hash differently")
	}
}

func TestMatchIsLongestCachedPrefix(t *testing.T) {
	ix := New(4, 100)
	sys := strings.Repeat("s", 16) // four blocks of shared system prompt
	ix.Record("e1", ix.Hashes(sys+"user question one"))
	if got := ix.Match("e1", ix.Hashes(sys+"another question")); got != 4 {
		t.Errorf("match = %d, want the 4 shared blocks", got)
	}
	if got := ix.Match("e2", ix.Hashes(sys)); got != 0 {
		t.Errorf("unknown endpoint matched %d blocks", got)
	}
	ix.Forget("e1")
	if ix.Match("e1", ix.Hashes(sys)) != 0 {
		t.Error("Forget kept blocks")
	}
}

func TestEvictionKeepsPrefixes(t *testing.T) {
	ix := New(4, 6)
	first := ix.Hashes(strings.Repeat("a", 16))  // 4 blocks
	second := ix.Hashes(strings.Repeat("b", 16)) // 4 more: capacity 6 forces 2 evictions
	ix.Record("e", first)
	ix.Record("e", second)
	if ix.Blocks("e") != 6 {
		t.Fatalf("blocks = %d, want capacity 6", ix.Blocks("e"))
	}
	if got := ix.Match("e", second); got != 4 {
		t.Errorf("recent sequence match = %d, want 4", got)
	}
	// The older sequence lost its tail, not its head.
	if got := ix.Match("e", first); got != 2 {
		t.Errorf("older sequence match = %d, want its first 2 blocks", got)
	}
}

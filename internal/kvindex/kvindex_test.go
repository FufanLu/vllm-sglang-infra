package kvindex

import (
	"testing"

	"github.com/FufanLu/vllm-sglang-infra/internal/kvevents"
)

func seq(from, n uint32) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = from + uint32(i)
	}
	return out
}

func stored(hashes []string, parent string, tokens []uint32, medium string) kvevents.Batch {
	return kvevents.Batch{Events: []kvevents.Event{{Type: kvevents.BlockStored, BlockHashes: hashes,
		ParentHash: parent, TokenIDs: tokens, BlockSize: 4, Medium: medium}}}
}

func TestStoredBlocksMatchTokenizedPrompt(t *testing.T) {
	ix := New()
	// The engine caches a 12-token prompt in two events: blocks 0-1, then
	// block 2 extending the cached prefix.
	ix.Apply("e", stored([]string{"h0", "h1"}, "", seq(100, 8), "GPU"))
	ix.Apply("e", stored([]string{"h2"}, "h1", seq(108, 4), "GPU"))

	prompt := append(seq(100, 12), 999, 998) // same 12 tokens plus a partial block
	if gpu, any := ix.Match("e", Hashes(prompt, ix.BlockSize("e"))); gpu != 3 || any != 3 {
		t.Errorf("match = %d/%d, want all 3 full blocks", gpu, any)
	}
	diverge := append(seq(100, 4), seq(500, 8)...)
	if gpu, _ := ix.Match("e", Hashes(diverge, 4)); gpu != 1 {
		t.Errorf("prompt diverging after block 0 matched %d blocks, want 1", gpu)
	}
}

func TestRemovalAndTiers(t *testing.T) {
	ix := New()
	ix.Apply("e", stored([]string{"h0", "h1", "h2"}, "", seq(0, 12), "GPU"))
	// Offloading copies blocks 1-2 to CPU, then the GPU evicts them.
	ix.Apply("e", stored([]string{"h1", "h2"}, "h0", seq(4, 8), "CPU"))
	ix.Apply("e", kvevents.Batch{Events: []kvevents.Event{{Type: kvevents.BlockRemoved, BlockHashes: []string{"h1", "h2"}, Medium: "GPU"}}})

	gpu, any := ix.Match("e", Hashes(seq(0, 12), 4))
	if gpu != 1 || any != 3 {
		t.Errorf("match = gpu %d any %d, want 1 on GPU and 3 counting the CPU tier", gpu, any)
	}
	if g, c, _ := ix.Stats("e"); g != 1 || c != 2 {
		t.Errorf("stats gpu %d cpu %d", g, c)
	}
	ix.Apply("e", kvevents.Batch{Events: []kvevents.Event{{Type: kvevents.AllBlocksCleared}}})
	if _, any := ix.Match("e", Hashes(seq(0, 12), 4)); any != 0 {
		t.Error("blocks survived AllBlocksCleared")
	}
}

func TestOrphansAreNotIndexed(t *testing.T) {
	ix := New()
	ix.Apply("e", stored([]string{"h5"}, "never-seen", seq(20, 4), "GPU"))
	if g, _, orphans := ix.Stats("e"); g != 0 || orphans != 1 {
		t.Errorf("gpu %d orphans %d: a block whose parent is unknown cannot be keyed", g, orphans)
	}
}

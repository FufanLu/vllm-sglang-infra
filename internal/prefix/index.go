// Package prefix is an approximate index of which endpoints hold which prompt
// prefixes in their KV cache.
//
// The gateway cannot see an engine's cache, so it learns from its own routing
// decisions: a request sent to an endpoint leaves its prompt cached there.
// Evictions are modelled with a per-endpoint LRU sized like the engine's cache.
// llm-d calls this approximate prefix-cache scoring; the precise alternative
// subscribes to the engine's KV-cache events.
package prefix

import (
	"container/list"
	"hash/fnv"
	"sync"
)

type Index struct {
	blockChars int
	capacity   int // blocks remembered per endpoint

	mu    sync.Mutex
	nodes map[string]*lru
}

// New returns an index that hashes prompts in blocks of blockChars bytes and
// remembers up to capacityBlocks blocks per endpoint.
func New(blockChars, capacityBlocks int) *Index {
	return &Index{blockChars: blockChars, capacity: capacityBlocks, nodes: make(map[string]*lru)}
}

// Hashes splits prompt into full blocks and returns one chained hash per
// block: the hash of block i covers the whole prompt up to the end of block
// i, so two prompts share hash i exactly when they share that prefix. A
// trailing partial block is ignored; engines cache only full blocks too.
func (ix *Index) Hashes(prompt string) []uint64 {
	n := len(prompt) / ix.blockChars
	if n == 0 {
		return nil
	}
	out := make([]uint64, n)
	h := fnv.New64a()
	for i := range n {
		h.Write([]byte(prompt[i*ix.blockChars : (i+1)*ix.blockChars]))
		out[i] = h.Sum64()
	}
	return out
}

// Match returns how many leading blocks of hashes are cached on key.
func (ix *Index) Match(key string, hashes []uint64) int {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	c := ix.nodes[key]
	if c == nil {
		return 0
	}
	n := 0
	for _, h := range hashes {
		if _, ok := c.items[h]; !ok {
			break
		}
		n++
	}
	return n
}

// Record marks hashes as cached on key. Blocks are touched from the last to
// the first so that, as in vLLM, the tail of a sequence is evicted before its
// head and a cached prefix stays a prefix.
func (ix *Index) Record(key string, hashes []uint64) {
	if len(hashes) == 0 {
		return
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	c := ix.nodes[key]
	if c == nil {
		c = &lru{order: list.New(), items: make(map[uint64]*list.Element)}
		ix.nodes[key] = c
	}
	for i := len(hashes) - 1; i >= 0; i-- {
		c.touch(hashes[i], ix.capacity)
	}
}

// Forget drops everything known about key.
func (ix *Index) Forget(key string) {
	ix.mu.Lock()
	delete(ix.nodes, key)
	ix.mu.Unlock()
}

// Blocks reports how many blocks are remembered for key.
func (ix *Index) Blocks(key string) int {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if c := ix.nodes[key]; c != nil {
		return c.order.Len()
	}
	return 0
}

type lru struct {
	order *list.List // front is most recently used
	items map[uint64]*list.Element
}

func (c *lru) touch(h uint64, capacity int) {
	if e, ok := c.items[h]; ok {
		c.order.MoveToFront(e)
		return
	}
	c.items[h] = c.order.PushFront(h)
	for c.order.Len() > capacity {
		old := c.order.Back()
		c.order.Remove(old)
		delete(c.items, old.Value.(uint64))
	}
}

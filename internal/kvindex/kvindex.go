// Package kvindex keeps an exact view of which KV-cache blocks each endpoint
// holds, in which tier, built from the engines' own KV-cache events.
//
// Engine block hashes are opaque to the gateway: vLLM hashes token IDs with
// SHA-256 and a per-deployment seed. So the index re-keys every stored block
// by its own chained hash of the block's token IDs, which the events carry.
// The gateway computes the same hash from a tokenized prompt (Hashes), and a
// prompt's blocks match exactly the blocks the engine would reuse.
package kvindex

import (
	"encoding/binary"
	"hash/fnv"
	"math"
	"sync"

	"github.com/FufanLu/vllm-sglang-infra/internal/kvevents"
)

// Tier bits; a block can be held in several tiers at once.
const (
	GPU uint8 = 1 << iota
	CPU
	Other // disk, object storage, ...
)

// tierOf returns the index of the medium's tier bit.
func tierOf(medium string) int {
	switch medium {
	case "", "GPU":
		return 0
	case "CPU":
		return 1
	}
	return 2
}

// held counts a block's announcements per tier. vLLM's offloading connector
// announces a block once per offloaded chunk that contains it, and removes
// it once per evicted chunk, so the offload tiers are reference-counted. The
// GPU prefix cache announces each block once.
type held [3]uint16

func (h held) tiers() uint8 {
	var t uint8
	for i, n := range h {
		if n > 0 {
			t |= 1 << i
		}
	}
	return t
}

type Index struct {
	mu  sync.Mutex
	eps map[string]*endpoint
}

type endpoint struct {
	blockSize int
	ours      map[string]uint64 // engine hash -> our hash
	blocks    map[uint64]held   // our hash -> tiers holding it
	orphans   int               // stored blocks we could not key
}

func New() *Index { return &Index{eps: map[string]*endpoint{}} }

// Apply folds a batch of events from the endpoint with key into the index.
func (ix *Index) Apply(key string, b kvevents.Batch) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ep := ix.eps[key]
	if ep == nil {
		ep = &endpoint{ours: map[string]uint64{}, blocks: map[uint64]held{}}
		ix.eps[key] = ep
	}
	for _, ev := range b.Events {
		switch ev.Type {
		case kvevents.BlockStored:
			ep.store(ev)
		case kvevents.BlockRemoved:
			tier := tierOf(ev.Medium)
			for _, eh := range ev.BlockHashes {
				h, ok := ep.ours[eh]
				if !ok {
					continue
				}
				c := ep.blocks[h]
				if c[tier] > 0 {
					c[tier]--
				}
				if c == (held{}) {
					delete(ep.blocks, h)
					delete(ep.ours, eh)
				} else {
					ep.blocks[h] = c
				}
			}
		case kvevents.AllBlocksCleared:
			*ep = endpoint{blockSize: ep.blockSize, ours: map[string]uint64{}, blocks: map[uint64]held{}}
		}
	}
}

func (ep *endpoint) store(ev kvevents.Event) {
	bs := ev.BlockSize
	if bs <= 0 || len(ev.TokenIDs) < bs*len(ev.BlockHashes) {
		// No token IDs to key the blocks by, as in the offloading
		// connector's placeholder events (self_describing_kv_events off).
		ep.orphans += len(ev.BlockHashes)
		return
	}
	ep.blockSize = bs
	var prev uint64
	if ev.ParentHash != "" {
		p, ok := ep.ours[ev.ParentHash]
		if !ok {
			ep.orphans += len(ev.BlockHashes) // missed events: these blocks cannot be chained
			return
		}
		prev = p
	}
	tier := tierOf(ev.Medium)
	for i, eh := range ev.BlockHashes {
		prev = chain(prev, ev.TokenIDs[i*bs:(i+1)*bs])
		ep.ours[eh] = prev
		c := ep.blocks[prev]
		if tier == 0 {
			c[0] = 1
		} else if c[tier] < math.MaxUint16 {
			c[tier]++
		}
		ep.blocks[prev] = c
	}
}

// Hashes returns the chained hash of each full block of tokens.
func Hashes(tokens []uint32, blockSize int) []uint64 {
	if blockSize <= 0 {
		return nil
	}
	out := make([]uint64, len(tokens)/blockSize)
	var prev uint64
	for i := range out {
		prev = chain(prev, tokens[i*blockSize:(i+1)*blockSize])
		out[i] = prev
	}
	return out
}

func chain(prev uint64, tokens []uint32) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], prev)
	h.Write(buf[:])
	for _, t := range tokens {
		binary.LittleEndian.PutUint32(buf[:4], t)
		h.Write(buf[:4])
	}
	return h.Sum64()
}

// Match reports how many leading blocks of hashes the endpoint holds: on
// GPU, and in any tier. A block counts in the GPU run only while every
// earlier block was on GPU too, since reuse stops at the first miss.
func (ix *Index) Match(key string, hashes []uint64) (gpu, any int) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ep := ix.eps[key]
	if ep == nil {
		return 0, 0
	}
	onGPU := true
	for _, h := range hashes {
		tiers := ep.blocks[h].tiers()
		if tiers == 0 {
			break
		}
		if onGPU && tiers&GPU != 0 {
			gpu++
		} else {
			onGPU = false
		}
		any++
	}
	return gpu, any
}

// BlockSize is the block size the endpoint's events reported, or 0 if no
// block has been stored yet.
func (ix *Index) BlockSize(key string) int {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if ep := ix.eps[key]; ep != nil {
		return ep.blockSize
	}
	return 0
}

// Stats reports blocks held per tier and orphaned blocks for the endpoint.
func (ix *Index) Stats(key string) (gpu, cpu, orphans int) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ep := ix.eps[key]
	if ep == nil {
		return 0, 0, 0
	}
	for _, c := range ep.blocks {
		t := c.tiers()
		if t&GPU != 0 {
			gpu++
		}
		if t&CPU != 0 {
			cpu++
		}
	}
	return gpu, cpu, ep.orphans
}

// Forget drops the endpoint.
func (ix *Index) Forget(key string) {
	ix.mu.Lock()
	delete(ix.eps, key)
	ix.mu.Unlock()
}

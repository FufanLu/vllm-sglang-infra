package kvevents

import (
	"testing"

	"github.com/vmihailenco/msgpack/v5"
)

func TestDecodeMapFormat(t *testing.T) {
	// vLLM 0.26: events are maps tagged by "type"; hashes may be bytes.
	payload, _ := msgpack.Marshal([]any{1.5, []any{
		map[string]any{"type": "BlockStored", "block_hashes": []any{[]byte{0xab, 0xcd}, uint64(7)},
			"parent_block_hash": nil, "token_ids": []any{1, 2, 3, 4, 5, 6, 7, 8}, "block_size": 4,
			"lora_id": nil, "medium": "GPU", "lora_name": nil},
		map[string]any{"type": "BlockRemoved", "block_hashes": []any{uint64(7)}, "medium": "CPU"},
		map[string]any{"type": "AllBlocksCleared"},
	}, 1})
	b, err := Decode(payload)
	if err != nil {
		t.Fatal(err)
	}
	if b.TS != 1.5 || b.Rank != 1 || len(b.Events) != 3 {
		t.Fatalf("batch = %+v", b)
	}
	s := b.Events[0]
	if s.Type != BlockStored || s.BlockHashes[0] != "abcd" || s.BlockHashes[1] != "7" || s.ParentHash != "" ||
		len(s.TokenIDs) != 8 || s.TokenIDs[7] != 8 || s.BlockSize != 4 || s.Medium != "GPU" {
		t.Errorf("stored = %+v", s)
	}
	if r := b.Events[1]; r.Type != BlockRemoved || r.BlockHashes[0] != "7" || r.Medium != "CPU" {
		t.Errorf("removed = %+v", r)
	}
	if b.Events[2].Type != AllBlocksCleared {
		t.Errorf("cleared = %+v", b.Events[2])
	}
}

func TestDecodeArrayFormat(t *testing.T) {
	// llm-d-inference-sim default: positional arrays, uint64 hashes, parent 0
	// meaning none.
	payload, _ := msgpack.Marshal([]any{2.0, []any{
		[]any{"BlockStored", []any{uint64(11), uint64(12)}, uint64(0), []any{9, 9, 9, 9, 8, 8, 8, 8}, 4, nil, "GPU", nil, nil},
		[]any{"BlockStored", []any{uint64(13)}, uint64(12), []any{7, 7, 7, 7}, 4, nil, nil, nil, nil},
		[]any{"BlockRemoved", []any{uint64(11)}, "GPU"},
	}})
	b, err := Decode(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Events) != 3 {
		t.Fatalf("events = %+v", b.Events)
	}
	if e := b.Events[0]; e.ParentHash != "" || e.BlockHashes[1] != "12" || e.BlockSize != 4 {
		t.Errorf("first = %+v", e)
	}
	if e := b.Events[1]; e.ParentHash != "12" || e.Medium != "GPU" {
		t.Errorf("second (nil medium means GPU) = %+v", e)
	}
	if e := b.Events[2]; e.Type != BlockRemoved || e.BlockHashes[0] != "11" {
		t.Errorf("removed = %+v", e)
	}
}

func TestDecodeRejectsGarbage(t *testing.T) {
	for _, p := range [][]byte{{0xc1}, mustPack(t, "x"), mustPack(t, []any{1.0, []any{[]any{"Nope"}}})} {
		if _, err := Decode(p); err == nil {
			t.Errorf("payload %x decoded", p)
		}
	}
}

func mustPack(t *testing.T, v any) []byte {
	b, err := msgpack.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestTopicAddress(t *testing.T) {
	for topic, want := range map[string]string{
		"kv@sim-0:8000@qwen3-8b-sim":     "sim-0:8000",
		"kv@10.0.0.1:8000@Qwen/Qwen3-8B": "10.0.0.1:8000",
		"kv@sim-2@qwen3-8b-sim":          "sim-2",
		"kv-events":                      "",
		"":                               "",
	} {
		if got := TopicAddress(topic); got != want {
			t.Errorf("TopicAddress(%q) = %q, want %q", topic, got, want)
		}
	}
}

func TestDecodeSGLangFormat(t *testing.T) {
	// SGLang 0.5.10 (disaggregation/kv_events.py): positional arrays with
	// trailing defaults omitted, signed int64 hashes, nil parent for a root
	// page, one page per event, and attn_dp_rank as the batch's third field.
	payload, _ := msgpack.Marshal([]any{3.0, []any{
		[]any{"BlockStored", []any{int64(-42)}, nil, []any{1, 2, 3, 4}, 4, nil, "GPU"},
		[]any{"BlockStored", []any{int64(7)}, int64(-42), []any{5, 6, 7, 8}, 4, nil, "GPU"},
		[]any{"BlockRemoved", []any{int64(7)}, "GPU"},
	}, nil})
	b, err := Decode(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Events) != 3 || b.Rank != 0 {
		t.Fatalf("batch = %+v", b)
	}
	first, second := b.Events[0], b.Events[1]
	if first.ParentHash != "" || len(first.TokenIDs) != 4 || first.BlockSize != 4 {
		t.Errorf("first = %+v", first)
	}
	if second.ParentHash != first.BlockHashes[0] {
		t.Errorf("second's parent %q should name the first block %q", second.ParentHash, first.BlockHashes[0])
	}
	if e := b.Events[2]; e.Type != BlockRemoved || e.BlockHashes[0] != second.BlockHashes[0] {
		t.Errorf("removed = %+v", e)
	}
}

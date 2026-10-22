// Package kvevents subscribes to the KV-cache events model servers publish
// over ZeroMQ — vLLM with --kv-events-config, llm-d-inference-sim with
// --enable-kvcache — and decodes them.
//
// Each message has three frames: topic, an 8-byte big-endian sequence
// number, and a msgpack event batch [ts, events, data_parallel_rank]. vLLM
// encodes each event as a map tagged by "type"; the simulator by default (and
// older vLLM) as a positional array tagged by its first element. Decode
// accepts both.
package kvevents

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"
)

const (
	BlockStored      = "BlockStored"
	BlockRemoved     = "BlockRemoved"
	AllBlocksCleared = "AllBlocksCleared"
)

type Event struct {
	Type string
	// BlockHashes are the engine's block hashes, normalized to strings:
	// integers in decimal, byte strings in hex.
	BlockHashes []string
	ParentHash  string   // BlockStored: hash of the cached block before these; "" if none
	TokenIDs    []uint32 // BlockStored: the tokens of the stored blocks, in order
	BlockSize   int      // BlockStored
	Medium      string   // GPU unless the event says otherwise (CPU for offloaded blocks)
}

type Batch struct {
	Seq    uint64
	TS     float64
	Rank   int
	Events []Event
}

// Decode parses one msgpack event batch.
func Decode(payload []byte) (Batch, error) {
	var raw any
	dec := msgpack.NewDecoder(bytes.NewReader(payload))
	dec.UseLooseInterfaceDecoding(true)
	if err := dec.Decode(&raw); err != nil {
		return Batch{}, err
	}
	arr, ok := raw.([]any)
	if !ok || len(arr) < 2 {
		return Batch{}, errors.New("event batch is not [ts, events, ...]")
	}
	var b Batch
	b.TS, _ = toFloat(arr[0])
	if len(arr) > 2 {
		r, _ := toInt(arr[2])
		b.Rank = int(r)
	}
	events, _ := arr[1].([]any)
	for _, e := range events {
		ev, err := decodeEvent(e)
		if err != nil {
			return b, err
		}
		b.Events = append(b.Events, ev)
	}
	return b, nil
}

// Field order of the positional encoding, after the tag.
var arrayFields = map[string][]string{
	BlockStored:      {"block_hashes", "parent_block_hash", "token_ids", "block_size", "lora_id", "medium", "lora_name", "extra_keys"},
	BlockRemoved:     {"block_hashes", "medium"},
	AllBlocksCleared: {},
}

func decodeEvent(v any) (Event, error) {
	fields := map[string]any{}
	var tag string
	switch e := v.(type) {
	case map[string]any:
		tag, _ = e["type"].(string)
		fields = e
	case []any:
		if len(e) == 0 {
			return Event{}, errors.New("empty event")
		}
		tag, _ = e[0].(string)
		for i, name := range arrayFields[tag] {
			if i+1 < len(e) {
				fields[name] = e[i+1]
			}
		}
	default:
		return Event{}, fmt.Errorf("event of type %T", v)
	}
	if _, known := arrayFields[tag]; !known {
		return Event{}, fmt.Errorf("unknown event type %q", tag)
	}
	ev := Event{Type: tag, Medium: "GPU"}
	if m, ok := fields["medium"].(string); ok && m != "" {
		ev.Medium = m
	}
	if hs, ok := fields["block_hashes"].([]any); ok {
		for _, h := range hs {
			ev.BlockHashes = append(ev.BlockHashes, hashString(h))
		}
	}
	if p := fields["parent_block_hash"]; p != nil {
		if s := hashString(p); s != "0" {
			ev.ParentHash = s
		}
	}
	if ts, ok := fields["token_ids"].([]any); ok {
		ev.TokenIDs = make([]uint32, len(ts))
		for i, t := range ts {
			n, _ := toInt(t)
			ev.TokenIDs[i] = uint32(n)
		}
	}
	if bs, ok := toInt(fields["block_size"]); ok {
		ev.BlockSize = int(bs)
	}
	return ev, nil
}

func hashString(v any) string {
	switch h := v.(type) {
	case []byte:
		return hex.EncodeToString(h)
	case string: // a byte-string hash some decoders surface as str
		return hex.EncodeToString([]byte(h))
	}
	if u, ok := v.(uint64); ok {
		return strconv.FormatUint(u, 10)
	}
	if n, ok := toInt(v); ok {
		return strconv.FormatUint(uint64(n), 10)
	}
	return fmt.Sprint(v)
}

func toInt(v any) (int64, bool) {
	switch n := v.(type) {
	case int64:
		return n, true
	case uint64:
		return int64(n), true
	case float64:
		return int64(n), true
	}
	return 0, false
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int64:
		return float64(n), true
	case uint64:
		return float64(n), true
	}
	return 0, false
}

// Subscribe connects to a publisher and calls handle with every batch and
// its topic until ctx ends, reconnecting after failures. topic filters by
// prefix; "" receives everything.
func Subscribe(ctx context.Context, endpoint, topic string, handle func(topic string, b Batch), log *slog.Logger) {
	for wait := time.Second; ctx.Err() == nil; wait = min(2*wait, 30*time.Second) {
		err := subscribeOnce(ctx, endpoint, topic, handle)
		if ctx.Err() != nil {
			return
		}
		log.Debug("kv events subscription dropped", "endpoint", endpoint, "err", err, "retry_in", wait.String())
		select {
		case <-time.After(wait):
		case <-ctx.Done():
		}
	}
}

func subscribeOnce(ctx context.Context, endpoint, topic string, handle func(string, Batch)) error {
	sub := zmq4.NewSub(ctx)
	defer sub.Close()
	if err := sub.Dial(endpoint); err != nil {
		return err
	}
	if err := sub.SetOption(zmq4.OptionSubscribe, topic); err != nil {
		return err
	}
	for {
		msg, err := sub.Recv()
		if err != nil {
			return err
		}
		if len(msg.Frames) < 3 {
			continue
		}
		b, err := Decode(msg.Frames[2])
		if err != nil {
			continue
		}
		if len(msg.Frames[1]) == 8 {
			b.Seq = binary.BigEndian.Uint64(msg.Frames[1])
		}
		handle(string(msg.Frames[0]), b)
	}
}

// TopicAddress extracts the address from a topic of the form
// kv@<host>:<port>@<model> (llm-d's convention) or kv@<host>@<model>, or ""
// for other topics.
func TopicAddress(topic string) string {
	rest, ok := strings.CutPrefix(topic, "kv@")
	if !ok {
		return ""
	}
	addr, _, ok := strings.Cut(rest, "@")
	if !ok {
		return ""
	}
	return addr
}

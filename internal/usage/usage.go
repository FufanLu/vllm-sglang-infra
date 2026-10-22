// Package usage records one line per request: who, what, where, how many
// tokens, how fast, and what it cost. Records are the basis for billing and
// for every per-tenant report.
package usage

import (
	"bufio"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Record struct {
	Time             time.Time `json:"time"`
	RequestID        string    `json:"request_id"`
	Tenant           string    `json:"tenant"`
	Model            string    `json:"model"`
	Backend          string    `json:"backend"`
	Route            string    `json:"route"`
	Status           string    `json:"status"`
	PromptTokens     int       `json:"prompt_tokens"`
	CachedTokens     int       `json:"cached_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	TTFTms           float64   `json:"ttft_ms"`
	DurationMs       float64   `json:"duration_ms"`
	QueueMs          float64   `json:"queue_ms"`
	Retries          int       `json:"retries"`
	External         bool      `json:"external"`
	CostUSD          float64   `json:"cost_usd"`
}

// Sink stores records. Write must not block the request path: sinks buffer
// and drop (counting the drop) when their backend cannot keep up.
type Sink interface {
	Write(Record)
	Close() error
}

// Discard drops every record.
type Discard struct{}

func (Discard) Write(Record) {}
func (Discard) Close() error { return nil }

// batcher runs flush on batches of records from a bounded queue, every
// interval or when a batch fills up.
type batcher struct {
	queue   chan Record
	flush   func([]Record) error
	log     *slog.Logger
	done    chan struct{}
	dropped sync.Once
}

func newBatcher(size int, interval time.Duration, flush func([]Record) error, log *slog.Logger) *batcher {
	b := &batcher{queue: make(chan Record, size*4), flush: flush, log: log, done: make(chan struct{})}
	go func() {
		defer close(b.done)
		tick := time.NewTicker(interval)
		defer tick.Stop()
		var batch []Record
		emit := func() {
			if len(batch) == 0 {
				return
			}
			if err := b.flush(batch); err != nil {
				b.log.Warn("usage flush failed", "records", len(batch), "err", err)
			}
			batch = batch[:0]
		}
		for {
			select {
			case r, ok := <-b.queue:
				if !ok {
					emit()
					return
				}
				if batch = append(batch, r); len(batch) >= size {
					emit()
				}
			case <-tick.C:
				emit()
			}
		}
	}()
	return b
}

func (b *batcher) Write(r Record) {
	select {
	case b.queue <- r:
	default:
		b.dropped.Do(func() { b.log.Warn("usage queue full; dropping records") })
	}
}

func (b *batcher) Close() error {
	close(b.queue)
	<-b.done
	return nil
}

// JSONL appends records to a file, one JSON object per line.
type JSONL struct {
	*batcher
	f *os.File
}

func NewJSONL(path string, log *slog.Logger) (*JSONL, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	w := bufio.NewWriter(f)
	j := &JSONL{f: f}
	j.batcher = newBatcher(256, time.Second, func(rs []Record) error {
		enc := json.NewEncoder(w)
		for _, r := range rs {
			if err := enc.Encode(r); err != nil {
				return err
			}
		}
		return w.Flush()
	}, log)
	return j, nil
}

func (j *JSONL) Close() error {
	j.batcher.Close()
	return j.f.Close()
}

// Postgres inserts records with COPY in batches.
type Postgres struct {
	*batcher
	pool *pgxpool.Pool
}

const schema = `
CREATE TABLE IF NOT EXISTS usage_records (
  time              timestamptz NOT NULL,
  request_id        text NOT NULL,
  tenant            text NOT NULL,
  model             text NOT NULL,
  backend           text NOT NULL,
  route             text NOT NULL,
  status            text NOT NULL,
  prompt_tokens     integer NOT NULL,
  cached_tokens     integer NOT NULL,
  completion_tokens integer NOT NULL,
  ttft_ms           double precision NOT NULL,
  duration_ms       double precision NOT NULL,
  queue_ms          double precision NOT NULL,
  retries           integer NOT NULL,
  external          boolean NOT NULL,
  cost_usd          double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_records_tenant_time ON usage_records (tenant, time);
`

var columns = []string{"time", "request_id", "tenant", "model", "backend", "route", "status",
	"prompt_tokens", "cached_tokens", "completion_tokens", "ttft_ms", "duration_ms", "queue_ms",
	"retries", "external", "cost_usd"}

func NewPostgres(ctx context.Context, dsn string, log *slog.Logger) (*Postgres, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, err
	}
	if _, err := pool.Exec(ctx, schema); err != nil {
		pool.Close()
		return nil, err
	}
	p := &Postgres{pool: pool}
	p.batcher = newBatcher(500, time.Second, func(rs []Record) error {
		rows := make([][]any, len(rs))
		for i, r := range rs {
			rows[i] = []any{r.Time, r.RequestID, r.Tenant, r.Model, r.Backend, r.Route, r.Status,
				r.PromptTokens, r.CachedTokens, r.CompletionTokens, r.TTFTms, r.DurationMs, r.QueueMs,
				r.Retries, r.External, r.CostUSD}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, err := pool.CopyFrom(ctx, pgx.Identifier{"usage_records"}, columns, pgx.CopyFromRows(rows))
		return err
	}, log)
	return p, nil
}

func (p *Postgres) Close() error {
	p.batcher.Close()
	p.pool.Close()
	return nil
}

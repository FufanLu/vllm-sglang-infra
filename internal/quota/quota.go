// Package quota enforces per-tenant token budgets.
//
// Budgets are in tokens, not requests: one request may be ten tokens or a
// hundred thousand. The output length is unknown when a request arrives, so
// the gateway reserves an estimate (prompt + max_tokens) up front and settles
// with the usage the engine reports, charging or refunding the difference.
//
// Each tenant has a token bucket holding at most one minute of budget and
// refilling continuously at TokensPerMinute/60 per second. A bucket may go
// negative when actual usage exceeds the reservation; the debt is repaid by
// refill before the tenant's next request is admitted.
package quota

import (
	"context"
	"math"
	"sync"
	"time"
)

type Limiter interface {
	// Reserve takes n tokens from the tenant's bucket. If the bucket holds
	// fewer than n, nothing is taken and retryAfter says how long until it
	// will hold n. A full bucket always admits, so a request larger than a
	// minute's budget is slow rather than impossible.
	Reserve(ctx context.Context, tenant string, perMinute, n int64) (ok bool, retryAfter time.Duration, err error)
	// Adjust charges delta more tokens (delta > 0) or refunds them (delta < 0).
	Adjust(ctx context.Context, tenant string, perMinute, delta int64) error
}

// Memory is a Limiter for a single gateway process.
type Memory struct {
	mu      sync.Mutex
	buckets map[string]*bucket
	now     func() time.Time
}

type bucket struct {
	tokens float64
	last   time.Time
}

func NewMemory() *Memory {
	return &Memory{buckets: map[string]*bucket{}, now: time.Now}
}

func (m *Memory) refill(tenant string, perMinute int64) *bucket {
	now := m.now()
	b, ok := m.buckets[tenant]
	if !ok {
		b = &bucket{tokens: float64(perMinute), last: now}
		m.buckets[tenant] = b
	}
	b.tokens = math.Min(float64(perMinute), b.tokens+now.Sub(b.last).Seconds()*float64(perMinute)/60)
	b.last = now
	return b
}

func (m *Memory) Reserve(_ context.Context, tenant string, perMinute, n int64) (bool, time.Duration, error) {
	if perMinute <= 0 {
		return true, 0, nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	b := m.refill(tenant, perMinute)
	if b.tokens >= float64(n) || b.tokens >= float64(perMinute) {
		b.tokens -= float64(n)
		return true, 0, nil
	}
	return false, waitFor(float64(n)-b.tokens, perMinute), nil
}

func (m *Memory) Adjust(_ context.Context, tenant string, perMinute, delta int64) error {
	if perMinute <= 0 || delta == 0 {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	b := m.refill(tenant, perMinute)
	b.tokens = math.Min(float64(perMinute), b.tokens-float64(delta))
	return nil
}

// waitFor is how long refill takes to produce missing tokens.
func waitFor(missing float64, perMinute int64) time.Duration {
	return time.Duration(math.Ceil(missing*60/float64(perMinute)*1000)) * time.Millisecond
}

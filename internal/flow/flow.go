// Package flow decides when a request may be dispatched to the pool.
//
// When the pool is saturated, requests wait in the gateway, in a queue per
// (priority band, tenant), instead of piling up in engine queues where
// nothing can reorder them. They leave in an order that protects interactive
// traffic and is fair across tenants:
//
//   - Bands are strict: interactive requests go before batch requests.
//   - Within a band, the tenant with the smallest virtual token counter goes
//     next. Each tenant's counter grows by the tokens it is served divided by
//     its weight (the Virtual Token Counter of "Fairness in Serving Large
//     Language Models", OSDI 2024), so a tenant sending long prompts cannot
//     starve the others. A tenant returning from idle is lifted to the
//     smallest active counter, so idleness does not bank credit.
//   - Batch requests may hold at most BatchShare of the slots, so a batch
//     flood never takes the slots interactive requests need.
package flow

import (
	"container/list"
	"context"
	"errors"
	"math"
	"sync"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/tenancy"
)

var (
	ErrQueueFull    = errors.New("flow control queue is full")
	ErrQueueTimeout = errors.New("timed out waiting for capacity")
)

type Config struct {
	// Capacity returns the number of requests the pool can run at once. It
	// is called on every decision, so it can follow endpoints coming and going.
	Capacity     func() int
	BatchShare   float64
	MaxQueued    [2]int           // per band, indexed by tenancy.Priority
	QueueTimeout [2]time.Duration // per band
}

type Controller struct {
	cfg Config

	mu       sync.Mutex
	inflight [2]int
	queued   [2]int
	queues   [2]map[string]*list.List // band -> tenant -> waiters, oldest first
	counters [2]map[string]float64
}

type waiter struct {
	tenant   string
	weight   float64
	cost     float64
	ready    chan struct{}
	admitted bool
}

// Ticket is a dispatch slot. Release it exactly once.
type Ticket struct {
	c        *Controller
	band     tenancy.Priority
	tenant   string
	weight   float64
	charged  float64
	released bool
	// Waited is how long the request queued before it was admitted.
	Waited time.Duration
}

func New(cfg Config) *Controller {
	c := &Controller{cfg: cfg}
	for b := range 2 {
		c.queues[b] = map[string]*list.List{}
		c.counters[b] = map[string]float64{}
	}
	return c
}

// Acquire returns when the request may be dispatched, when its band's queue
// timeout passes, or when ctx ends. cost is the request's estimated tokens.
func (c *Controller) Acquire(ctx context.Context, band tenancy.Priority, tenant string, weight, cost float64) (*Ticket, error) {
	start := time.Now()
	c.mu.Lock()
	if t := c.tryAdmitLocked(band, tenant, weight, cost); t != nil {
		c.mu.Unlock()
		return t, nil
	}
	if c.queued[band] >= c.cfg.MaxQueued[band] {
		c.mu.Unlock()
		return nil, ErrQueueFull
	}
	w := &waiter{tenant: tenant, weight: weight, cost: cost, ready: make(chan struct{})}
	q := c.queues[band][tenant]
	if q == nil || q.Len() == 0 {
		c.liftLocked(band, tenant)
	}
	if q == nil {
		q = list.New()
		c.queues[band][tenant] = q
	}
	elem := q.PushBack(w)
	c.queued[band]++
	c.mu.Unlock()

	timer := time.NewTimer(c.cfg.QueueTimeout[band])
	defer timer.Stop()
	var err error
	select {
	case <-w.ready:
		return &Ticket{c: c, band: band, tenant: tenant, weight: weight, charged: cost, Waited: time.Since(start)}, nil
	case <-ctx.Done():
		err = ctx.Err()
	case <-timer.C:
		err = ErrQueueTimeout
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if w.admitted {
		// Admitted while we were giving up: hand the slot back.
		c.releaseLocked(band, tenant, weight, cost, 0)
		return nil, err
	}
	q.Remove(elem)
	c.queued[band]--
	return nil, err
}

// TryAcquire admits the request only if it would not have to wait.
func (c *Controller) TryAcquire(band tenancy.Priority, tenant string, weight, cost float64) *Ticket {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.tryAdmitLocked(band, tenant, weight, cost)
}

func (c *Controller) tryAdmitLocked(band tenancy.Priority, tenant string, weight, cost float64) *Ticket {
	// Never overtake waiters of the same or a higher band.
	for b := band; b <= tenancy.Interactive; b++ {
		if c.queued[b] > 0 {
			return nil
		}
	}
	if !c.canAdmitLocked(band) {
		return nil
	}
	c.liftLocked(band, tenant)
	c.admitLocked(band, tenant, weight, cost)
	return &Ticket{c: c, band: band, tenant: tenant, weight: weight, charged: cost}
}

func (c *Controller) canAdmitLocked(band tenancy.Priority) bool {
	capacity := c.cfg.Capacity()
	if c.inflight[0]+c.inflight[1] >= capacity {
		return false
	}
	if band == tenancy.Batch {
		limit := max(1, int(math.Floor(c.cfg.BatchShare*float64(capacity))))
		return c.inflight[tenancy.Batch] < limit
	}
	return true
}

func (c *Controller) admitLocked(band tenancy.Priority, tenant string, weight, cost float64) {
	c.inflight[band]++
	c.counters[band][tenant] += cost / weight
}

// liftLocked raises a newly active tenant's counter to the smallest counter
// among tenants already waiting in the band.
func (c *Controller) liftLocked(band tenancy.Priority, tenant string) {
	lowest, any := math.Inf(1), false
	for t, q := range c.queues[band] {
		if q.Len() > 0 && t != tenant {
			lowest, any = math.Min(lowest, c.counters[band][t]), true
		}
	}
	if any && c.counters[band][tenant] < lowest {
		c.counters[band][tenant] = lowest
	}
}

// dispatchLocked admits waiters while capacity allows, highest band first,
// least-served tenant first within a band.
func (c *Controller) dispatchLocked() {
	for {
		band := tenancy.Priority(-1)
		for b := tenancy.Interactive; b >= tenancy.Batch; b-- {
			if c.queued[b] > 0 {
				if c.canAdmitLocked(b) {
					band = b
				}
				break // strict priority: lower bands wait while a higher one is queued
			}
		}
		if band < 0 {
			// Interactive is empty or blocked; batch may still fit.
			if c.queued[tenancy.Interactive] == 0 && c.queued[tenancy.Batch] > 0 && c.canAdmitLocked(tenancy.Batch) {
				band = tenancy.Batch
			} else {
				return
			}
		}
		var next string
		best := math.Inf(1)
		for t, q := range c.queues[band] {
			if q.Len() > 0 && c.counters[band][t] < best {
				next, best = t, c.counters[band][t]
			}
		}
		q := c.queues[band][next]
		w := q.Remove(q.Front()).(*waiter)
		c.queued[band]--
		w.admitted = true
		c.admitLocked(band, w.tenant, w.weight, w.cost)
		close(w.ready)
	}
}

func (c *Controller) releaseLocked(band tenancy.Priority, tenant string, weight, charged, actual float64) {
	c.inflight[band]--
	if actual > 0 {
		c.counters[band][tenant] += (actual - charged) / weight
	}
	c.dispatchLocked()
}

// Release frees the slot. actual is the tokens the request really used
// (0 if unknown), which corrects the tenant's counter.
func (t *Ticket) Release(actual float64) {
	if t == nil || t.released {
		return
	}
	t.released = true
	t.c.mu.Lock()
	defer t.c.mu.Unlock()
	t.c.releaseLocked(t.band, t.tenant, t.weight, t.charged, actual)
}

// Poke re-runs dispatch; call it when capacity may have grown, such as when
// an endpoint becomes healthy.
func (c *Controller) Poke() {
	c.mu.Lock()
	c.dispatchLocked()
	c.mu.Unlock()
}

// Stats reports in-flight and queued requests per band.
func (c *Controller) Stats() (inflight, queued [2]int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.inflight, c.queued
}

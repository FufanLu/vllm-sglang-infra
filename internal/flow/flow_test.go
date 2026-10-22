package flow

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/tenancy"
)

const (
	batch       = tenancy.Batch
	interactive = tenancy.Interactive
)

func newController(capacity int) *Controller {
	return New(Config{
		Capacity:     func() int { return capacity },
		BatchShare:   0.5,
		MaxQueued:    [2]int{100, 100},
		QueueTimeout: [2]time.Duration{5 * time.Second, 5 * time.Second},
	})
}

// acquireAsync starts an Acquire and reports the order in which waiters are
// admitted.
func acquireAsync(t *testing.T, c *Controller, band tenancy.Priority, tenant string, order chan<- string, wg *sync.WaitGroup) {
	t.Helper()
	wg.Add(1)
	go func() {
		defer wg.Done()
		tk, err := c.Acquire(context.Background(), band, tenant, 1, 100)
		if err != nil {
			t.Error(err)
			return
		}
		order <- tenant
		tk.Release(100)
	}()
}

// waitQueued blocks until n requests are queued in band.
func waitQueued(t *testing.T, c *Controller, band tenancy.Priority, n int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, q := c.Stats(); q[band] == n {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("queued in band %d never reached %d", band, n)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestInteractiveGoesFirst(t *testing.T) {
	c := newController(1)
	hold, _ := c.Acquire(context.Background(), batch, "bulk", 1, 100) // fill the pool
	order := make(chan string, 4)
	var wg sync.WaitGroup
	acquireAsync(t, c, batch, "bulk", order, &wg)
	waitQueued(t, c, batch, 1)
	acquireAsync(t, c, interactive, "chat", order, &wg)
	waitQueued(t, c, interactive, 1)
	hold.Release(100)
	wg.Wait()
	close(order)
	if first := <-order; first != "chat" {
		t.Errorf("first admitted = %s, want the interactive request despite arriving later", first)
	}
}

func TestBatchShareReservesSlots(t *testing.T) {
	c := newController(4) // batch may hold floor(0.5*4) = 2
	var held []*Ticket
	for range 2 {
		tk := c.TryAcquire(batch, "bulk", 1, 100)
		if tk == nil {
			t.Fatal("batch refused below its share")
		}
		held = append(held, tk)
	}
	if c.TryAcquire(batch, "bulk", 1, 100) != nil {
		t.Error("batch exceeded its share of capacity")
	}
	if c.TryAcquire(interactive, "chat", 1, 100) == nil {
		t.Error("interactive refused while slots were reserved for it")
	}
	for _, tk := range held {
		tk.Release(0)
	}
}

func TestFairnessByTokens(t *testing.T) {
	c := newController(1)
	hold, _ := c.Acquire(context.Background(), interactive, "x", 1, 1)
	// Both tenants stay backlogged. heavy's requests cost 10x light's, so
	// light should be admitted about ten times for each heavy request.
	order := make(chan string, 32)
	var wg sync.WaitGroup
	enqueue := func(tenant string, cost float64, n int) {
		for range n {
			wg.Add(1)
			go func() {
				defer wg.Done()
				tk, err := c.Acquire(context.Background(), interactive, tenant, 1, cost)
				if err != nil {
					t.Error(err)
					return
				}
				order <- tenant
				tk.Release(cost)
			}()
		}
	}
	enqueue("heavy", 1000, 4)
	waitQueued(t, c, interactive, 4)
	enqueue("light", 100, 12)
	waitQueued(t, c, interactive, 16)
	hold.Release(1)
	wg.Wait()
	close(order)
	heavyInFirst := 0
	i := 0
	for tenant := range order {
		if i < 8 && tenant == "heavy" {
			heavyInFirst++
		}
		i++
	}
	if heavyInFirst > 2 {
		t.Errorf("heavy got %d of the first 8 slots; token fairness should give most to light", heavyInFirst)
	}
}

func TestIdleTenantIsLifted(t *testing.T) {
	c := newController(1)
	hold, _ := c.Acquire(context.Background(), interactive, "x", 1, 1)
	c.mu.Lock()
	c.counters[interactive]["busy"] = 5000
	c.mu.Unlock()
	var wg sync.WaitGroup
	order := make(chan string, 4)
	acquireAsync(t, c, interactive, "busy", order, &wg)
	waitQueued(t, c, interactive, 1)
	acquireAsync(t, c, interactive, "newcomer", order, &wg) // counter 0, lifted to 5000
	waitQueued(t, c, interactive, 2)
	c.mu.Lock()
	lifted := c.counters[interactive]["newcomer"]
	c.mu.Unlock()
	hold.Release(1)
	wg.Wait()
	if lifted != 5000 {
		t.Errorf("newcomer counter = %v, want lifted to 5000", lifted)
	}
}

func TestQueueLimitsAndTimeout(t *testing.T) {
	c := New(Config{
		Capacity:     func() int { return 1 },
		BatchShare:   1,
		MaxQueued:    [2]int{1, 1},
		QueueTimeout: [2]time.Duration{50 * time.Millisecond, 50 * time.Millisecond},
	})
	hold, _ := c.Acquire(context.Background(), interactive, "a", 1, 1)
	defer hold.Release(1)

	errc := make(chan error, 1)
	go func() {
		_, err := c.Acquire(context.Background(), interactive, "b", 1, 1)
		errc <- err
	}()
	waitQueued(t, c, interactive, 1)
	if _, err := c.Acquire(context.Background(), interactive, "c", 1, 1); !errors.Is(err, ErrQueueFull) {
		t.Errorf("over the queue limit: err = %v, want ErrQueueFull", err)
	}
	if err := <-errc; !errors.Is(err, ErrQueueTimeout) {
		t.Errorf("err = %v, want ErrQueueTimeout", err)
	}
	if _, q := c.Stats(); q[interactive] != 0 {
		t.Errorf("timed-out waiter still queued: %v", q)
	}
}

func TestCancelledWaiterLeavesQueue(t *testing.T) {
	c := newController(1)
	hold, _ := c.Acquire(context.Background(), interactive, "a", 1, 1)
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := c.Acquire(ctx, interactive, "b", 1, 1)
		errc <- err
	}()
	waitQueued(t, c, interactive, 1)
	cancel()
	if err := <-errc; !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v", err)
	}
	hold.Release(1)
	if in, q := c.Stats(); in[interactive] != 0 || q[interactive] != 0 {
		t.Errorf("slots leaked: inflight %v queued %v", in, q)
	}
}

func TestCapacityGrowthAdmitsWaiters(t *testing.T) {
	capacity := 0
	var mu sync.Mutex
	c := New(Config{
		Capacity:     func() int { mu.Lock(); defer mu.Unlock(); return capacity },
		BatchShare:   1,
		MaxQueued:    [2]int{10, 10},
		QueueTimeout: [2]time.Duration{time.Second, time.Second},
	})
	errc := make(chan error, 1)
	go func() {
		tk, err := c.Acquire(context.Background(), interactive, "a", 1, 1)
		if err == nil {
			tk.Release(1)
		}
		errc <- err
	}()
	waitQueued(t, c, interactive, 1)
	mu.Lock()
	capacity = 1 // an endpoint became healthy
	mu.Unlock()
	c.Poke()
	if err := <-errc; err != nil {
		t.Errorf("waiter not admitted after capacity grew: %v", err)
	}
}

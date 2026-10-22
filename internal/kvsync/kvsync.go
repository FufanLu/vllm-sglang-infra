// Package kvsync keeps KV-event subscriptions open for registered endpoints
// that publish events (label kv_events), feeding the precise KV index.
//
// Endpoints usually publish on their own address: vLLM binds its publisher,
// one subscription per replica. Several endpoints may also share an address,
// such as a relay that fans in publishers that dial out (llm-d-inference-sim
// does); events are then attributed by their topic, kv@<host>:<port>@<model>,
// matched against each endpoint's URL. Subscriptions follow the registry.
package kvsync

import (
	"context"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/kvevents"
	"github.com/FufanLu/vllm-sglang-infra/internal/kvindex"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
)

// Label names the endpoint label holding the event publisher's address.
const Label = "kv_events"

// route maps a subscription's messages to endpoint keys.
type route struct {
	mu     sync.RWMutex
	keys   []string          // endpoints on this address, sorted
	byAddr map[string]string // host:port -> key, for shared addresses
}

func (r *route) target(topic string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if len(r.keys) == 1 {
		return r.keys[0]
	}
	return r.byAddr[kvevents.TopicAddress(topic)]
}

// Watch runs until ctx ends.
func Watch(ctx context.Context, reg *backend.Registry, ix *kvindex.Index, log *slog.Logger) {
	type sub struct {
		cancel context.CancelFunc
		route  *route
	}
	subs := map[string]*sub{} // publisher address -> subscription
	known := map[string]bool{}
	sync := func() {
		want := map[string][]*backend.Endpoint{}
		for _, ep := range reg.List() {
			if addr := ep.Label(Label); addr != "" {
				want[addr] = append(want[addr], ep)
			}
		}
		live := map[string]bool{}
		for addr, eps := range want {
			keys := make([]string, 0, len(eps))
			byAddr := map[string]string{}
			hosts := map[string]int{}
			for _, ep := range eps {
				hosts[ep.URL.Hostname()]++
			}
			for _, ep := range eps {
				k := scheduler.EndpointKey(ep)
				keys = append(keys, k)
				byAddr[ep.URL.Host] = k
				// Some publishers name only the host (llm-d-inference-sim
				// v0.9 uses kv@<pod-ip>@<model>); that is unambiguous only
				// when one endpoint lives on the host.
				if hosts[ep.URL.Hostname()] == 1 {
					byAddr[ep.URL.Hostname()] = k
				}
				live[k] = true
			}
			slices.Sort(keys)
			s, ok := subs[addr]
			if !ok {
				s = &sub{route: &route{}}
				sctx, cancel := context.WithCancel(ctx)
				s.cancel = cancel
				subs[addr] = s
				rt := s.route
				log.Info("subscribing to kv events", "publisher", addr, "endpoints", strings.Join(keys, ","))
				go kvevents.Subscribe(sctx, addr, "", func(topic string, b kvevents.Batch) {
					if k := rt.target(topic); k != "" {
						ix.Apply(k, b)
					}
				}, log)
			}
			s.route.mu.Lock()
			s.route.keys, s.route.byAddr = keys, byAddr
			s.route.mu.Unlock()
		}
		for addr, s := range subs {
			if _, ok := want[addr]; !ok {
				s.cancel()
				delete(subs, addr)
			}
		}
		for k := range known {
			if !live[k] {
				ix.Forget(k)
			}
		}
		known = live
	}
	sync()
	t := time.NewTicker(2 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			sync()
		}
	}
}

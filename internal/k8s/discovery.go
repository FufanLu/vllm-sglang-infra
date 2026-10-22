package k8s

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
)

// SourceLabel marks endpoints this package manages, so discovery never
// removes endpoints that came from static config or the admin API.
const SourceLabel = "source"

// PodLister is the part of Client discovery needs; tests fake it.
type PodLister interface {
	Pods(ctx context.Context, selector string) ([]Pod, error)
}

// Discovery keeps the registry in step with model-server pods: a ready pod
// is an endpoint at http://<pod IP>:<Port>, labelled with the pod's labels;
// a pod that is gone, unready, or terminating is removed. Annotations
// prefixed vsi.io/ become labels too (for example vsi.io/kv_events:
// pod:5557, the pod's own KV-event publisher).
type Discovery struct {
	Client   PodLister
	Selector string
	Port     int
	Engine   backend.Engine
	Interval time.Duration
	Registry *backend.Registry
	// OnAdd runs for each newly added endpoint (the gateway probes it).
	OnAdd func(*backend.Endpoint)
	Log   *slog.Logger
}

func (d *Discovery) Run(ctx context.Context) {
	t := time.NewTicker(d.Interval)
	defer t.Stop()
	for {
		if err := d.Sync(ctx); err != nil {
			d.Log.Warn("pod discovery failed", "selector", d.Selector, "err", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// Sync reconciles once.
func (d *Discovery) Sync(ctx context.Context) error {
	pods, err := d.Client.Pods(ctx, d.Selector)
	if err != nil {
		return err
	}
	want := map[string]bool{}
	for _, p := range pods {
		if !p.Running || !p.Ready || p.Deleting || p.IP == "" {
			continue
		}
		labels := maps.Clone(p.Labels)
		if labels == nil {
			labels = map[string]string{}
		}
		for k, v := range p.Annotations {
			if name, ok := cutPrefix(k, "vsi.io/"); ok {
				// "pod:<port>" names a port on the pod itself, which a
				// pod template cannot spell out (vsi.io/kv_events: pod:5557).
				if port, ok := cutPrefix(v, "pod:"); ok {
					v = fmt.Sprintf("tcp://%s:%s", p.IP, port)
				}
				labels[name] = v
			}
		}
		labels[SourceLabel] = "kubernetes"
		want[p.Name] = true
		existed := d.Registry.Get(p.Name) != nil
		ep, err := d.Registry.Upsert(p.Name, fmt.Sprintf("http://%s:%d", p.IP, d.Port), d.Engine, labels)
		if err != nil {
			return err
		}
		if !existed && d.OnAdd != nil {
			d.Log.Info("pod discovered", "endpoint", ep.Name, "url", ep.URL.String())
			d.OnAdd(ep)
		}
	}
	for _, ep := range d.Registry.List() {
		if ep.Label(SourceLabel) == "kubernetes" && !want[ep.Name] {
			d.Log.Info("pod gone", "endpoint", ep.Name)
			d.Registry.Remove(ep.Name)
		}
	}
	return nil
}

func cutPrefix(s, prefix string) (string, bool) {
	if len(s) > len(prefix) && s[:len(prefix)] == prefix {
		return s[len(prefix):], true
	}
	return "", false
}

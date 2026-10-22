package backend

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// Monitor keeps endpoint state fresh. It probes /health, scrapes /metrics, and
// discovers the served models from /v1/models.
type Monitor struct {
	Registry         *Registry
	Client           *http.Client
	HealthInterval   time.Duration
	ScrapeInterval   time.Duration
	Timeout          time.Duration // per probe or scrape
	FailureThreshold int
	Log              *slog.Logger
}

// Run probes and scrapes every endpoint once, then on each interval, until ctx
// is done.
func (m *Monitor) Run(ctx context.Context) {
	m.ProbeAll(ctx)
	m.ScrapeAll(ctx)
	health := time.NewTicker(m.HealthInterval)
	scrape := time.NewTicker(m.ScrapeInterval)
	defer health.Stop()
	defer scrape.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-health.C:
			m.ProbeAll(ctx)
		case <-scrape.C:
			m.ScrapeAll(ctx)
		}
	}
}

// ProbeAll health-checks every endpoint concurrently and waits for all probes.
func (m *Monitor) ProbeAll(ctx context.Context) {
	m.each(ctx, m.Check)
}

// ScrapeAll reads engine metrics from every endpoint concurrently.
func (m *Monitor) ScrapeAll(ctx context.Context) {
	m.each(ctx, m.Scrape)
}

func (m *Monitor) each(ctx context.Context, fn func(context.Context, *Endpoint)) {
	var wg sync.WaitGroup
	for _, ep := range m.Registry.List() {
		wg.Go(func() { fn(ctx, ep) })
	}
	wg.Wait()
}

// externalRetry is how long an external provider stays out of rotation after
// a failure. Providers have no health endpoint, so it is simply retried.
const externalRetry = 30 * time.Second

// Check probes one endpoint. When it is healthy but its models are unknown,
// Check also asks it which models it serves.
func (m *Monitor) Check(ctx context.Context, ep *Endpoint) {
	if ep.IsExternal() {
		if d := ep.DownFor(); d > externalRetry && ep.RecordProbe(true, m.FailureThreshold) {
			m.Log.Info("external provider back in rotation", "endpoint", ep.Name, "after", d.Round(time.Second).String())
		}
		return
	}
	err := m.get(ctx, ep, "/health", nil)
	if ep.RecordProbe(err == nil, m.FailureThreshold) {
		if err == nil {
			m.Log.Info("endpoint up", "endpoint", ep.Name, "url", ep.URL.String())
		} else {
			m.Log.Warn("endpoint down", "endpoint", ep.Name, "err", err)
		}
	}
	if err == nil && len(ep.Models()) == 0 {
		m.discoverModels(ctx, ep)
	}
}

func (m *Monitor) discoverModels(ctx context.Context, ep *Endpoint) {
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	err := m.get(ctx, ep, "/v1/models", func(body io.Reader) error {
		return json.NewDecoder(body).Decode(&list)
	})
	if err != nil {
		m.Log.Debug("model discovery failed", "endpoint", ep.Name, "err", err)
		return
	}
	models := make([]string, 0, len(list.Data))
	for _, d := range list.Data {
		models = append(models, d.ID)
	}
	ep.SetModels(models)
}

// Scrape reads one endpoint's /metrics. On failure the previous snapshot is
// kept; scorers judge freshness by Snapshot.ScrapedAt.
func (m *Monitor) Scrape(ctx context.Context, ep *Endpoint) {
	if ep.IsExternal() {
		return
	}
	inflight := ep.Inflight()
	err := m.get(ctx, ep, "/metrics", func(body io.Reader) error {
		samples, err := parsePromText(body)
		if err != nil {
			return err
		}
		snap, ok := extractSnapshot(ep.Engine, samples)
		if !ok {
			return fmt.Errorf("no %s scheduler metrics found", ep.Engine)
		}
		snap.ScrapedAt, snap.GatewayInflight = time.Now(), inflight
		ep.SetSnapshot(snap)
		return nil
	})
	if err != nil {
		m.Log.Debug("scrape failed", "endpoint", ep.Name, "err", err)
	}
}

// get issues a GET against the endpoint and hands a 200 response body to read.
func (m *Monitor) get(ctx context.Context, ep *Endpoint, path string, read func(io.Reader) error) error {
	ctx, cancel := context.WithTimeout(ctx, m.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ep.URL.String()+path, nil)
	if err != nil {
		return err
	}
	resp, err := m.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		return fmt.Errorf("GET %s: status %d", path, resp.StatusCode)
	}
	if read == nil {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		return nil
	}
	return read(io.LimitReader(resp.Body, 8<<20))
}

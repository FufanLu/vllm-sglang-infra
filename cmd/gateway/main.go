// Command gateway is the inference gateway: it authenticates tenants, enforces
// their token budgets, schedules OpenAI-compatible requests onto vLLM / SGLang
// endpoints (or external providers on overflow), and proxies the responses.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/autoscale"
	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/config"
	"github.com/FufanLu/vllm-sglang-infra/internal/flow"
	"github.com/FufanLu/vllm-sglang-infra/internal/gateway"
	"github.com/FufanLu/vllm-sglang-infra/internal/k8s"
	"github.com/FufanLu/vllm-sglang-infra/internal/kvindex"
	"github.com/FufanLu/vllm-sglang-infra/internal/kvsync"
	"github.com/FufanLu/vllm-sglang-infra/internal/metrics"
	"github.com/FufanLu/vllm-sglang-infra/internal/quota"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
	"github.com/FufanLu/vllm-sglang-infra/internal/tenancy"
	"github.com/FufanLu/vllm-sglang-infra/internal/tokenize"
	"github.com/FufanLu/vllm-sglang-infra/internal/usage"
)

func main() {
	cfgPath := flag.String("config", "", "path to the gateway YAML config (defaults apply when empty)")
	debug := flag.Bool("debug", false, "log at debug level")
	hashKey := flag.Bool("hash-key", false, "read an API key from stdin, print its hash for tenancy config, and exit")
	flag.Parse()

	if *hashKey {
		key, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		fmt.Println(tenancy.HashKey(strings.TrimSpace(key)))
		return
	}
	level := slog.LevelInfo
	if *debug {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	if err := run(*cfgPath, log); err != nil {
		log.Error("gateway failed", "err", err)
		os.Exit(1)
	}
}

func run(cfgPath string, log *slog.Logger) error {
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	reg := backend.NewRegistry()
	for _, b := range cfg.Backends {
		spec, err := backendSpec(b)
		if err != nil {
			return fmt.Errorf("backend %s: %w", b.Name, err)
		}
		if _, err := reg.UpsertSpec(spec); err != nil {
			return fmt.Errorf("backend %s: %w", b.Name, err)
		}
	}
	kvIndex := kvindex.New()
	go kvsync.Watch(ctx, reg, kvIndex, log)
	schedOpts := cfg.SchedulerOptions()
	schedOpts.KVIndex = kvIndex
	sched, err := scheduler.New(cfg.Scheduler.Policy, schedOpts)
	if err != nil {
		return err
	}
	tenants, err := buildTenants(cfg.Tenancy)
	if err != nil {
		return err
	}
	limiter, err := buildQuota(ctx, cfg.Quota, log)
	if err != nil {
		return err
	}
	sink, err := buildUsage(ctx, cfg.Usage, log)
	if err != nil {
		return err
	}
	defer sink.Close()

	var flowCtl *flow.Controller
	var flowStats metrics.FlowStats
	if cfg.Flow.Enabled {
		per := cfg.Flow.MaxInflightPerEndpoint
		flowCtl = flow.New(flow.Config{
			Capacity:     func() int { return per * servingEndpoints(reg) },
			BatchShare:   cfg.Flow.BatchShare,
			MaxQueued:    [2]int{cfg.Flow.MaxQueued.Batch, cfg.Flow.MaxQueued.Interactive},
			QueueTimeout: [2]time.Duration{cfg.Flow.QueueTimeout.Batch, cfg.Flow.QueueTimeout.Interactive},
		})
		flowStats = flowCtl.Stats
		go func() { // capacity grows when endpoints become healthy
			t := time.NewTicker(200 * time.Millisecond)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					flowCtl.Poke()
				}
			}
		}()
	}

	mon := &backend.Monitor{
		Registry:         reg,
		Client:           &http.Client{Transport: newTransport()},
		HealthInterval:   cfg.Health.Interval,
		ScrapeInterval:   cfg.Scrape.Interval,
		Timeout:          max(cfg.Health.Timeout, cfg.Scrape.Timeout),
		FailureThreshold: cfg.Health.FailureThreshold,
		Log:              log,
	}
	go mon.Run(ctx)

	m := metrics.New(reg, flowStats)
	m.RegisterKVIndex(reg, func(ep *backend.Endpoint) (int, int, int) { return kvIndex.Stats(scheduler.EndpointKey(ep)) })
	srv := gateway.New(gateway.Options{
		Registry:         reg,
		Scheduler:        sched,
		SchedulerOptions: schedOpts,
		Metrics:          m,
		Client:           &http.Client{Transport: newTransport()}, // no Timeout: generations stream for minutes
		Log:              log,
		MaxRetries:       cfg.Proxy.MaxRetries,
		MaxBodyBytes:     cfg.Proxy.MaxBodyBytes,
		AdminToken:       cfg.Admin.Token,
		OnRegister:       func(ep *backend.Endpoint) { go mon.Check(ctx, ep) },
		Tenants:          tenants,
		Quota:            limiter,
		Usage:            sink,
		Flow:             flowCtl,
		PD:               cfg.PD.Enabled,
		DefaultMaxTokens: cfg.Quota.DefaultMaxTokens,
		Tokenizer:        &tokenize.Client{HTTP: &http.Client{Transport: newTransport()}},
	})
	var kube *k8s.Client
	if cfg.Discovery.Kubernetes.Enabled || (cfg.Autoscale.Enabled && cfg.Autoscale.Provider == "kubernetes") {
		if kube, err = k8s.InCluster(); err != nil {
			return fmt.Errorf("kubernetes: %w", err)
		}
	}
	if kd := cfg.Discovery.Kubernetes; kd.Enabled {
		engine, _ := backend.ParseEngine(kd.Engine)
		d := &k8s.Discovery{Client: kube, Selector: kd.Selector, Port: kd.Port, Engine: engine,
			Interval: max(kd.Interval, time.Second), Registry: reg, Log: log,
			OnAdd: func(ep *backend.Endpoint) { go mon.Check(ctx, ep) }}
		go d.Run(ctx)
	}
	if cfg.Autoscale.Enabled {
		a, err := buildAutoscaler(cfg.Autoscale, reg, mon, flowCtl, kube, ctx, log)
		if err != nil {
			return err
		}
		go a.Run(ctx)
	}
	hs := &http.Server{Addr: cfg.Listen, Handler: srv.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		log.Info("shutting down", "grace", cfg.ShutdownGrace.String())
		sctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
		defer cancel()
		hs.Shutdown(sctx)
	}()

	log.Info("gateway listening", "addr", cfg.Listen, "policy", sched.Policy, "backends", len(cfg.Backends),
		"tenants", len(cfg.Tenancy.Tenants), "quota", cfg.Quota.Store, "usage", cfg.Usage.Sink,
		"flow_control", cfg.Flow.Enabled, "pd", cfg.PD.Enabled, "autoscale", cfg.Autoscale.Enabled,
		"admin_api", cfg.Admin.Token != "")
	if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func buildAutoscaler(c config.Autoscale, reg *backend.Registry, mon *backend.Monitor, f *flow.Controller,
	kube *k8s.Client, ctx context.Context, log *slog.Logger) (*autoscale.Autoscaler, error) {
	demand := func() (int, int) {
		if f != nil {
			in, q := f.Stats()
			return in[0] + in[1], q[0] + q[1]
		}
		n := 0
		for _, ep := range reg.List() {
			if !ep.IsExternal() {
				n += int(ep.Inflight())
			}
		}
		return n, 0
	}
	var p autoscale.Provider
	switch c.Provider {
	case "slurm":
		p = &autoscale.Slurm{Script: c.Slurm.Script, Args: c.Slurm.Args, JobName: cmpOr(c.Slurm.JobName, "vsi-replica")}
	case "kubernetes":
		p = &autoscale.Kubernetes{Client: kube, Deployment: c.Kubernetes.Deployment, Selector: c.Kubernetes.Selector}
	case "process":
		p = &autoscale.Process{
			Command: c.Process.Command,
			Register: func(name, url string) error {
				ep, err := reg.Upsert(name, url, backend.EngineVLLM, map[string]string{"autoscaled": "true"})
				if err == nil {
					go func() { // probe until the new process answers
						for range 50 {
							if mon.Check(ctx, ep); ep.Healthy() {
								return
							}
							time.Sleep(100 * time.Millisecond)
						}
					}()
				}
				return err
			},
			Deregister: func(name string) { reg.Remove(name) },
			Inflight: func(name string) int64 {
				if ep := reg.Get(name); ep != nil {
					return ep.Inflight()
				}
				return 0
			},
		}
	default:
		return nil, fmt.Errorf("unknown autoscale provider %q", c.Provider)
	}
	return autoscale.New(autoscale.Config{Min: c.Min, Max: c.Max, TargetPerReplica: c.TargetPerReplica,
		ScaleDownAfter: c.ScaleDownAfter, HandoverLead: c.HandoverLead, Interval: c.Interval}, p, demand, log), nil
}

func cmpOr(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

// servingEndpoints counts healthy local endpoints that decode: the slots
// flow control hands out. Prefill-only endpoints do not hold a request for
// its whole lifetime, and external providers are overflow, not capacity.
func servingEndpoints(reg *backend.Registry) int {
	n := 0
	for _, ep := range reg.List() {
		if ep.Healthy() && !ep.IsExternal() && ep.Label("role") != "prefill" {
			n++
		}
	}
	return n
}

func backendSpec(b config.Backend) (backend.Spec, error) {
	engine, _ := backend.ParseEngine(b.Engine) // validated by config.Load
	spec := backend.Spec{Name: b.Name, URL: b.URL, Engine: engine, Labels: b.Labels, Kind: backend.KindLocal}
	if b.Kind == "external" {
		key := os.Getenv(b.APIKeyEnv)
		if key == "" {
			return spec, fmt.Errorf("environment variable %s (api_key_env) is empty", b.APIKeyEnv)
		}
		spec.Kind = backend.KindExternal
		spec.External = &backend.External{APIKey: key, Models: b.Models, PriceInput: b.PriceInput,
			PriceOutput: b.PriceOutput, MaxInflight: max(b.MaxInflight, 1)}
	}
	return spec, nil
}

func buildTenants(t config.Tenancy) (*tenancy.Directory, error) {
	if len(t.Tenants) == 0 {
		return tenancy.NewOpen(), nil
	}
	tenants := make([]tenancy.Tenant, 0, len(t.Tenants))
	hashes := map[string][]string{}
	for _, c := range t.Tenants {
		p, _ := tenancy.ParsePriority(c.Priority) // validated by config.Load
		tenants = append(tenants, tenancy.Tenant{Name: c.Name, Priority: p, TokensPerMinute: c.TokensPerMinute,
			OnPremOnly: c.OnPremOnly, Weight: c.Weight})
		hashes[c.Name] = c.APIKeyHashes
	}
	return tenancy.New(tenants, hashes)
}

func buildQuota(ctx context.Context, q config.Quota, log *slog.Logger) (quota.Limiter, error) {
	if q.Store != "redis" {
		return quota.NewMemory(), nil
	}
	r, err := quota.NewRedis(q.RedisURL, "vsi:quota:")
	if err != nil {
		return nil, err
	}
	return r, retry(ctx, log, "redis", func(ctx context.Context) error { return r.Ping(ctx) })
}

func buildUsage(ctx context.Context, u config.Usage, log *slog.Logger) (usage.Sink, error) {
	switch u.Sink {
	case "jsonl":
		return usage.NewJSONL(u.Path, log)
	case "postgres":
		var p *usage.Postgres
		err := retry(ctx, log, "postgres", func(ctx context.Context) (err error) {
			p, err = usage.NewPostgres(ctx, u.PostgresDSN, log)
			return err
		})
		return p, err
	}
	return usage.Discard{}, nil
}

// retry calls fn until it succeeds or 30 seconds pass. Dependencies started
// alongside the gateway (by compose, Kubernetes, or a Slurm job) are often
// not ready the instant it starts.
func retry(ctx context.Context, log *slog.Logger, what string, fn func(context.Context) error) error {
	deadline := time.Now().Add(30 * time.Second)
	for wait := 250 * time.Millisecond; ; wait = min(2*wait, 4*time.Second) {
		actx, cancel := context.WithTimeout(ctx, 5*time.Second)
		err := fn(actx)
		cancel()
		if err == nil {
			return nil
		}
		if time.Now().After(deadline) || ctx.Err() != nil {
			return fmt.Errorf("%s: %w", what, err)
		}
		log.Warn("dependency not ready; retrying", "dependency", what, "err", err, "in", wait.String())
		select {
		case <-time.After(wait):
		case <-ctx.Done():
		}
	}
}

// newTransport is the HTTP transport for traffic to model servers. It never
// uses an HTTP proxy: HPC login environments and laptops often export
// http_proxy, and model traffic must stay on the cluster network. (External
// providers are reached directly too; set up egress explicitly if needed.)
func newTransport() *http.Transport {
	return &http.Transport{
		Proxy:               nil,
		DialContext:         (&net.Dialer{Timeout: 3 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:        1024,
		MaxIdleConnsPerHost: 256,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true,
	}
}

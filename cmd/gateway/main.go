// Command gateway is the inference gateway: it schedules OpenAI-compatible
// requests onto vLLM / SGLang endpoints and proxies the responses.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/config"
	"github.com/FufanLu/vllm-sglang-infra/internal/gateway"
	"github.com/FufanLu/vllm-sglang-infra/internal/metrics"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
)

func main() {
	cfgPath := flag.String("config", "", "path to the gateway YAML config (defaults apply when empty)")
	debug := flag.Bool("debug", false, "log at debug level")
	flag.Parse()

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
	reg := backend.NewRegistry()
	for _, b := range cfg.Backends {
		engine, _ := backend.ParseEngine(b.Engine) // validated by config.Load
		if _, err := reg.Upsert(b.Name, b.URL, engine, b.Labels); err != nil {
			return fmt.Errorf("backend %s: %w", b.Name, err)
		}
	}
	sched, err := scheduler.New(cfg.Scheduler.Policy, cfg.Scheduler.Weights)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

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

	srv := gateway.New(gateway.Options{
		Registry:     reg,
		Scheduler:    sched,
		Metrics:      metrics.New(reg),
		Client:       &http.Client{Transport: newTransport()}, // no Timeout: generations stream for minutes
		Log:          log,
		MaxRetries:   cfg.Proxy.MaxRetries,
		MaxBodyBytes: cfg.Proxy.MaxBodyBytes,
		AdminToken:   cfg.Admin.Token,
		OnRegister:   func(ep *backend.Endpoint) { go mon.Check(ctx, ep) },
	})
	hs := &http.Server{Addr: cfg.Listen, Handler: srv.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		log.Info("shutting down", "grace", cfg.ShutdownGrace.String())
		sctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
		defer cancel()
		hs.Shutdown(sctx)
	}()

	log.Info("gateway listening", "addr", cfg.Listen, "policy", sched.Policy,
		"backends", len(cfg.Backends), "admin_api", cfg.Admin.Token != "")
	if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// newTransport is the HTTP transport for traffic to model servers. It never
// uses an HTTP proxy: HPC login environments and laptops often export
// http_proxy, and model traffic must stay on the cluster network.
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

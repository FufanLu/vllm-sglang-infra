// Package config loads the gateway's YAML configuration.
package config

import (
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
)

type Config struct {
	Listen        string        `yaml:"listen"`
	ShutdownGrace time.Duration `yaml:"shutdown_grace"`
	Backends      []Backend     `yaml:"backends"`
	Scheduler     Scheduler     `yaml:"scheduler"`
	Health        Health        `yaml:"health"`
	Scrape        Scrape        `yaml:"scrape"`
	Proxy         Proxy         `yaml:"proxy"`
	Admin         Admin         `yaml:"admin"`
}

// Backend is a statically configured endpoint. Endpoints can also register
// at runtime through the admin API.
type Backend struct {
	Name   string            `yaml:"name"`
	URL    string            `yaml:"url"`
	Engine string            `yaml:"engine"` // vllm (default) or sglang
	Labels map[string]string `yaml:"labels"`
}

type Scheduler struct {
	Policy  string             `yaml:"policy"`  // see scheduler.Policies()
	Weights map[string]float64 `yaml:"weights"` // overrides the policy's scorer weights
	// QueueSaturation (prefix-aware) is the load at which the queue score
	// reaches 0; it sets how much busier a cache-holding endpoint may be.
	QueueSaturation float64 `yaml:"queue_saturation"`
	Prefix          Prefix  `yaml:"prefix"`
}

// Prefix sizes the approximate prefix index of the prefix-aware policy.
type Prefix struct {
	BlockChars     int `yaml:"block_chars"`     // prompt characters per hashed block
	CapacityBlocks int `yaml:"capacity_blocks"` // blocks remembered per endpoint (its cache size)
}

// Options converts the scheduler section for scheduler.New.
func (s Scheduler) Options() scheduler.Options {
	return scheduler.Options{
		Weights:              s.Weights,
		QueueSaturation:      s.QueueSaturation,
		PrefixBlockChars:     s.Prefix.BlockChars,
		PrefixCapacityBlocks: s.Prefix.CapacityBlocks,
	}
}

type Health struct {
	Interval         time.Duration `yaml:"interval"`
	Timeout          time.Duration `yaml:"timeout"`
	FailureThreshold int           `yaml:"failure_threshold"`
}

type Scrape struct {
	Interval time.Duration `yaml:"interval"`
	Timeout  time.Duration `yaml:"timeout"`
}

type Proxy struct {
	// MaxRetries is how many other backends a request may be re-sent to when
	// the chosen one refuses the connection or answers 502/503/504 before any
	// response byte reaches the client.
	MaxRetries   int   `yaml:"max_retries"`
	MaxBodyBytes int64 `yaml:"max_body_bytes"`
}

type Admin struct {
	// Token guards the /admin API. Empty disables it. The VSI_ADMIN_TOKEN
	// environment variable overrides the file so the secret can stay out of it.
	Token string `yaml:"token"`
}

func Default() Config {
	return Config{
		Listen:        ":8080",
		ShutdownGrace: 30 * time.Second,
		Scheduler:     Scheduler{Policy: "least-load"},
		Health:        Health{Interval: 2 * time.Second, Timeout: time.Second, FailureThreshold: 2},
		Scrape:        Scrape{Interval: 500 * time.Millisecond, Timeout: time.Second},
		Proxy:         Proxy{MaxRetries: 2, MaxBodyBytes: 16 << 20},
	}
}

// Load reads the YAML file at path over the defaults. An empty path yields
// the defaults. Unknown keys are rejected so typos fail loudly.
func Load(path string) (Config, error) {
	cfg := Default()
	if path != "" {
		f, err := os.Open(path)
		if err != nil {
			return cfg, err
		}
		defer f.Close()
		dec := yaml.NewDecoder(f)
		dec.KnownFields(true)
		if err := dec.Decode(&cfg); err != nil && !errors.Is(err, io.EOF) {
			return cfg, fmt.Errorf("%s: %w", path, err)
		}
	}
	if tok := os.Getenv("VSI_ADMIN_TOKEN"); tok != "" {
		cfg.Admin.Token = tok
	}
	if p := os.Getenv("VSI_SCHEDULER_POLICY"); p != "" {
		cfg.Scheduler.Policy = p
	}
	return cfg, cfg.Validate()
}

var nameRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$`)

// ValidName reports whether s can name a backend. Names appear in URLs and
// metric labels, so they are kept to a conservative character set.
func ValidName(s string) bool { return nameRE.MatchString(s) }

func (c *Config) Validate() error {
	var errs []error
	if c.Listen == "" {
		errs = append(errs, errors.New("listen is empty"))
	}
	seen := map[string]bool{}
	for i, b := range c.Backends {
		switch {
		case !ValidName(b.Name):
			errs = append(errs, fmt.Errorf("backends[%d]: invalid name %q", i, b.Name))
		case seen[b.Name]:
			errs = append(errs, fmt.Errorf("backends[%d]: duplicate name %q", i, b.Name))
		}
		seen[b.Name] = true
		if _, err := backend.ParseURL(b.URL); err != nil {
			errs = append(errs, fmt.Errorf("backends[%d]: %w", i, err))
		}
		if _, err := backend.ParseEngine(b.Engine); err != nil {
			errs = append(errs, fmt.Errorf("backends[%d]: %w", i, err))
		}
	}
	if _, err := scheduler.New(c.Scheduler.Policy, c.Scheduler.Options()); err != nil {
		errs = append(errs, err)
	}
	for name, d := range map[string]time.Duration{
		"health.interval": c.Health.Interval, "health.timeout": c.Health.Timeout,
		"scrape.interval": c.Scrape.Interval, "scrape.timeout": c.Scrape.Timeout,
	} {
		if d <= 0 {
			errs = append(errs, fmt.Errorf("%s must be positive", name))
		}
	}
	if c.Health.FailureThreshold < 1 {
		errs = append(errs, errors.New("health.failure_threshold must be at least 1"))
	}
	if c.Proxy.MaxRetries < 0 {
		errs = append(errs, errors.New("proxy.max_retries must not be negative"))
	}
	if c.Proxy.MaxBodyBytes <= 0 {
		errs = append(errs, errors.New("proxy.max_body_bytes must be positive"))
	}
	return errors.Join(errs...)
}

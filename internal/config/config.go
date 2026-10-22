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
	"github.com/FufanLu/vllm-sglang-infra/internal/kvindex"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
	"github.com/FufanLu/vllm-sglang-infra/internal/tenancy"
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
	Tenancy       Tenancy       `yaml:"tenancy"`
	Quota         Quota         `yaml:"quota"`
	Usage         Usage         `yaml:"usage"`
	Flow          Flow          `yaml:"flow"`
	PD            PD            `yaml:"pd"`
	Autoscale     Autoscale     `yaml:"autoscale"`
	Discovery     Discovery     `yaml:"discovery"`
}

// Discovery finds model-server endpoints by themselves, beyond static
// config and self-registration.
type Discovery struct {
	Kubernetes struct {
		Enabled  bool          `yaml:"enabled"`
		Selector string        `yaml:"selector"` // pod label selector, e.g. app=vllm
		Port     int           `yaml:"port"`     // model server port in the pod
		Engine   string        `yaml:"engine"`
		Interval time.Duration `yaml:"interval"`
	} `yaml:"kubernetes"`
}

// Backend is a statically configured endpoint. Endpoints can also register
// at runtime through the admin API.
type Backend struct {
	Name   string            `yaml:"name"`
	URL    string            `yaml:"url"`
	Engine string            `yaml:"engine"` // vllm (default) or sglang
	Labels map[string]string `yaml:"labels"` // e.g. role: prefill | decode
	// Kind is local (default) or external. External backends are
	// OpenAI-compatible providers used for overflow; their URL is the API
	// base, ending in /v1.
	Kind        string            `yaml:"kind"`
	APIKeyEnv   string            `yaml:"api_key_env"`  // external: environment variable holding the key
	Models      map[string]string `yaml:"models"`       // external: gateway model -> provider model
	PriceInput  float64           `yaml:"price_input"`  // external: USD per million prompt tokens
	PriceOutput float64           `yaml:"price_output"` // external: USD per million completion tokens
	MaxInflight int               `yaml:"max_inflight"` // external: concurrent requests allowed
}

// Tenancy lists the tenants. With none, the gateway is open and every
// request belongs to one default tenant.
type Tenancy struct {
	Tenants []Tenant `yaml:"tenants"`
}

type Tenant struct {
	Name string `yaml:"name"`
	// APIKeyHashes are "sha256:<hex>" digests of the tenant's keys; generate
	// one with `gateway -hash-key`.
	APIKeyHashes    []string `yaml:"api_key_hashes"`
	Priority        string   `yaml:"priority"` // interactive (default) or batch
	TokensPerMinute int64    `yaml:"tokens_per_minute"`
	OnPremOnly      bool     `yaml:"on_prem_only"`
	Weight          float64  `yaml:"weight"`
}

type Quota struct {
	Store    string `yaml:"store"`     // memory (default) or redis
	RedisURL string `yaml:"redis_url"` // or VSI_REDIS_URL
	// DefaultMaxTokens is the output length reserved for requests that do
	// not set max_tokens.
	DefaultMaxTokens int `yaml:"default_max_tokens"`
}

type Usage struct {
	Sink        string `yaml:"sink"`         // none (default), jsonl, or postgres
	Path        string `yaml:"path"`         // jsonl
	PostgresDSN string `yaml:"postgres_dsn"` // postgres; or VSI_POSTGRES_DSN
}

// Flow configures gateway-side queueing.
type Flow struct {
	Enabled bool `yaml:"enabled"`
	// MaxInflightPerEndpoint is each local endpoint's share of pool
	// capacity, for flow control and for deciding when to overflow.
	MaxInflightPerEndpoint int        `yaml:"max_inflight_per_endpoint"`
	BatchShare             float64    `yaml:"batch_share"` // fraction of capacity batch traffic may use
	MaxQueued              PerBand    `yaml:"max_queued"`
	QueueTimeout           PerBandDur `yaml:"queue_timeout"`
}

type PerBand struct {
	Interactive int `yaml:"interactive"`
	Batch       int `yaml:"batch"`
}

type PerBandDur struct {
	Interactive time.Duration `yaml:"interactive"`
	Batch       time.Duration `yaml:"batch"`
}

// PD enables prefill/decode disaggregation over endpoints labelled
// role: prefill and role: decode.
type PD struct {
	Enabled bool `yaml:"enabled"`
}

// Autoscale sizes the replica pool from demand (requests in flight plus
// queued) and replaces Slurm replicas before their walltime.
type Autoscale struct {
	Enabled          bool          `yaml:"enabled"`
	Provider         string        `yaml:"provider"` // slurm or process
	Min              int           `yaml:"min"`
	Max              int           `yaml:"max"`
	TargetPerReplica int           `yaml:"target_inflight_per_replica"`
	ScaleDownAfter   time.Duration `yaml:"scale_down_after"`
	HandoverLead     time.Duration `yaml:"handover_lead"`
	Interval         time.Duration `yaml:"interval"`
	Slurm            struct {
		Script  string   `yaml:"script"`
		Args    []string `yaml:"args"`
		JobName string   `yaml:"job_name"`
	} `yaml:"slurm"`
	Process struct {
		Command []string `yaml:"command"` // "{port}" is replaced by a free port
	} `yaml:"process"`
	Kubernetes struct {
		Deployment string `yaml:"deployment"`
		Selector   string `yaml:"selector"`
	} `yaml:"kubernetes"`
}

type Scheduler struct {
	Policy  string             `yaml:"policy"`  // see scheduler.Policies()
	Weights map[string]float64 `yaml:"weights"` // overrides the policy's scorer weights
	// QueueSaturation (prefix-aware) is the load at which the queue score
	// reaches 0; it sets how much busier a cache-holding endpoint may be.
	QueueSaturation float64 `yaml:"queue_saturation"`
	Prefix          Prefix  `yaml:"prefix"`
	// CPUWeight (precise-prefix-aware) values a prefix block held only in
	// CPU memory relative to one on GPU.
	CPUWeight float64 `yaml:"cpu_weight"`
}

// Prefix sizes the approximate prefix index of the prefix-aware policy.
type Prefix struct {
	BlockChars     int `yaml:"block_chars"`     // prompt characters per hashed block
	CapacityBlocks int `yaml:"capacity_blocks"` // blocks remembered per endpoint (its cache size)
}

// SchedulerOptions converts the scheduler section for scheduler.New.
func (c *Config) SchedulerOptions() scheduler.Options {
	return scheduler.Options{
		Weights:              c.Scheduler.Weights,
		QueueSaturation:      c.Scheduler.QueueSaturation,
		PrefixBlockChars:     c.Scheduler.Prefix.BlockChars,
		PrefixCapacityBlocks: c.Scheduler.Prefix.CapacityBlocks,
		Saturation:           c.Flow.MaxInflightPerEndpoint,
		CPUWeight:            c.Scheduler.CPUWeight,
		KVIndex:              validationIndex,
	}
}

// validationIndex lets Validate build a precise-prefix-aware scheduler;
// the gateway replaces it with the live index.
var validationIndex = kvindex.New()

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
		Quota:         Quota{Store: "memory", DefaultMaxTokens: 512},
		Usage:         Usage{Sink: "none"},
		Flow: Flow{
			MaxInflightPerEndpoint: 32,
			BatchShare:             0.75,
			MaxQueued:              PerBand{Interactive: 1000, Batch: 10000},
			QueueTimeout:           PerBandDur{Interactive: 30 * time.Second, Batch: 10 * time.Minute},
		},
		Autoscale: Autoscale{Provider: "slurm", Min: 1, Max: 4, TargetPerReplica: 16,
			ScaleDownAfter: 5 * time.Minute, HandoverLead: 20 * time.Minute, Interval: 15 * time.Second},
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
	if u := os.Getenv("VSI_REDIS_URL"); u != "" {
		cfg.Quota.RedisURL = u
	}
	if d := os.Getenv("VSI_POSTGRES_DSN"); d != "" {
		cfg.Usage.PostgresDSN = d
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
		switch b.Kind {
		case "", "local":
		case "external":
			if b.APIKeyEnv == "" || len(b.Models) == 0 {
				errs = append(errs, fmt.Errorf("backends[%d]: external backends need api_key_env and models", i))
			}
		default:
			errs = append(errs, fmt.Errorf("backends[%d]: kind must be local or external", i))
		}
	}
	tenants := map[string]bool{}
	for i, t := range c.Tenancy.Tenants {
		if !ValidName(t.Name) || tenants[t.Name] {
			errs = append(errs, fmt.Errorf("tenancy.tenants[%d]: invalid or duplicate name %q", i, t.Name))
		}
		tenants[t.Name] = true
		if _, err := tenancy.ParsePriority(t.Priority); err != nil {
			errs = append(errs, fmt.Errorf("tenancy.tenants[%d]: %w", i, err))
		}
		if len(t.APIKeyHashes) == 0 {
			errs = append(errs, fmt.Errorf("tenancy.tenants[%d]: no api_key_hashes", i))
		}
	}
	switch c.Quota.Store {
	case "memory":
	case "redis":
		if c.Quota.RedisURL == "" {
			errs = append(errs, errors.New("quota.store redis needs quota.redis_url or VSI_REDIS_URL"))
		}
	default:
		errs = append(errs, errors.New("quota.store must be memory or redis"))
	}
	switch c.Usage.Sink {
	case "none":
	case "jsonl":
		if c.Usage.Path == "" {
			errs = append(errs, errors.New("usage.sink jsonl needs usage.path"))
		}
	case "postgres":
		if c.Usage.PostgresDSN == "" {
			errs = append(errs, errors.New("usage.sink postgres needs usage.postgres_dsn or VSI_POSTGRES_DSN"))
		}
	default:
		errs = append(errs, errors.New("usage.sink must be none, jsonl, or postgres"))
	}
	f := c.Flow
	if f.MaxInflightPerEndpoint < 1 || f.BatchShare <= 0 || f.BatchShare > 1 ||
		f.MaxQueued.Interactive < 0 || f.MaxQueued.Batch < 0 || f.QueueTimeout.Interactive <= 0 || f.QueueTimeout.Batch <= 0 {
		errs = append(errs, errors.New("flow: max_inflight_per_endpoint >= 1, 0 < batch_share <= 1, queue limits >= 0, timeouts > 0"))
	}
	if a := c.Autoscale; a.Enabled {
		switch {
		case a.Min < 0 || a.Max < max(a.Min, 1) || a.TargetPerReplica < 1 || a.Interval <= 0:
			errs = append(errs, errors.New("autoscale: 0 <= min <= max, max >= 1, target_inflight_per_replica >= 1, interval > 0"))
		case a.Provider == "slurm" && a.Slurm.Script == "":
			errs = append(errs, errors.New("autoscale.slurm.script is required"))
		case a.Provider == "process" && len(a.Process.Command) == 0:
			errs = append(errs, errors.New("autoscale.process.command is required"))
		case a.Provider == "kubernetes" && (a.Kubernetes.Deployment == "" || a.Kubernetes.Selector == ""):
			errs = append(errs, errors.New("autoscale.kubernetes needs deployment and selector"))
		case a.Provider != "slurm" && a.Provider != "process" && a.Provider != "kubernetes":
			errs = append(errs, errors.New("autoscale.provider must be slurm, kubernetes, or process"))
		}
	}
	if k := c.Discovery.Kubernetes; k.Enabled {
		if k.Selector == "" || k.Port <= 0 {
			errs = append(errs, errors.New("discovery.kubernetes needs selector and port"))
		}
		if _, err := backend.ParseEngine(k.Engine); err != nil {
			errs = append(errs, fmt.Errorf("discovery.kubernetes: %w", err))
		}
	}
	if _, err := scheduler.New(c.Scheduler.Policy, c.SchedulerOptions()); err != nil {
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

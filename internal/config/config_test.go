package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func write(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "gateway.yaml")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadOverridesDefaults(t *testing.T) {
	cfg, err := Load(write(t, `
listen: ":9000"
scheduler:
  policy: round-robin
health:
  interval: 5s
backends:
  - name: replica-a
    url: http://d4052:8000
  - name: replica-b
    url: http://d4053:8000
    engine: sglang
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Listen != ":9000" || cfg.Scheduler.Policy != "round-robin" || cfg.Health.Interval != 5*time.Second {
		t.Errorf("overrides not applied: %+v", cfg)
	}
	if cfg.Health.Timeout != time.Second || cfg.Proxy.MaxRetries != 2 {
		t.Errorf("defaults lost: %+v", cfg)
	}
	if len(cfg.Backends) != 2 || cfg.Backends[1].Engine != "sglang" {
		t.Errorf("backends = %+v", cfg.Backends)
	}
}

func TestLoadRejectsBadConfig(t *testing.T) {
	for name, body := range map[string]string{
		"unknown key":    "listn: \":80\"\n",
		"duplicate name": "backends:\n- {name: a, url: 'http://x:1'}\n- {name: a, url: 'http://y:1'}\n",
		"bad url":        "backends:\n- {name: a, url: 'x:1'}\n",
		"bad engine":     "backends:\n- {name: a, url: 'http://x:1', engine: tgi}\n",
		"bad policy":     "scheduler: {policy: random}\n",
		"bad name":       "backends:\n- {name: 'a/b', url: 'http://x:1'}\n",
	} {
		if _, err := Load(write(t, body)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestAdminTokenFromEnv(t *testing.T) {
	t.Setenv("VSI_ADMIN_TOKEN", "s3cret")
	cfg, err := Load(write(t, "admin: {token: from-file}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Admin.Token != "s3cret" {
		t.Errorf("token = %q, want the environment value", cfg.Admin.Token)
	}
}

func TestEmptyPathIsDefaults(t *testing.T) {
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(cfg.Listen, ":") || cfg.Scheduler.Policy != "least-load" {
		t.Errorf("defaults = %+v", cfg)
	}
}

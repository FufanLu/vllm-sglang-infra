// Package tenancy maps API keys to tenants. Keys are stored only as SHA-256
// hashes, so the config file and logs never hold a usable credential.
package tenancy

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// Priority is a tenant's traffic class. Interactive traffic is served before
// batch traffic when the pool is saturated.
type Priority int

const (
	Batch Priority = iota
	Interactive
)

func ParsePriority(s string) (Priority, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "interactive":
		return Interactive, nil
	case "batch":
		return Batch, nil
	}
	return 0, fmt.Errorf("unknown priority %q (want interactive or batch)", s)
}

func (p Priority) String() string {
	if p == Batch {
		return "batch"
	}
	return "interactive"
}

type Tenant struct {
	Name     string
	Priority Priority
	// TokensPerMinute is the token budget (prompt + completion). Zero means
	// unlimited.
	TokensPerMinute int64
	// OnPremOnly keeps the tenant's requests on the cluster: they are never
	// overflowed to an external provider.
	OnPremOnly bool
	// Weight is the tenant's share relative to others of the same priority
	// when flow control has to choose whose request goes next.
	Weight float64
}

var ErrUnauthorized = errors.New("missing or invalid API key")

// Directory authenticates requests. With no tenants configured it is open:
// every request belongs to the default tenant.
type Directory struct {
	byHash   map[string]*Tenant
	fallback *Tenant
}

// Anonymous is the tenant of every request when tenancy is disabled.
var Anonymous = &Tenant{Name: "default", Priority: Interactive, Weight: 1}

// NewOpen returns a directory that accepts every request as Anonymous.
func NewOpen() *Directory { return &Directory{fallback: Anonymous} }

// New builds a directory from tenants and their key hashes. A hash may
// belong to only one tenant.
func New(tenants []Tenant, keyHashes map[string][]string) (*Directory, error) {
	d := &Directory{byHash: map[string]*Tenant{}}
	for i := range tenants {
		t := &tenants[i]
		if t.Weight <= 0 {
			t.Weight = 1
		}
		for _, h := range keyHashes[t.Name] {
			h = strings.ToLower(strings.TrimSpace(h))
			if !validHash(h) {
				return nil, fmt.Errorf("tenant %s: key hash %q is not sha256:<64 hex digits>", t.Name, h)
			}
			if other, dup := d.byHash[h]; dup {
				return nil, fmt.Errorf("key hash used by both %s and %s", other.Name, t.Name)
			}
			d.byHash[h] = t
		}
	}
	return d, nil
}

// Authenticate resolves an Authorization header ("Bearer <key>") to a tenant.
func (d *Directory) Authenticate(header string) (*Tenant, error) {
	if d.fallback != nil {
		return d.fallback, nil
	}
	key, ok := strings.CutPrefix(header, "Bearer ")
	if !ok || key == "" {
		return nil, ErrUnauthorized
	}
	if t, ok := d.byHash[HashKey(key)]; ok {
		return t, nil
	}
	return nil, ErrUnauthorized
}

// HashKey returns the stored form of an API key.
func HashKey(key string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(key)))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func validHash(h string) bool {
	hexPart, ok := strings.CutPrefix(h, "sha256:")
	if !ok || len(hexPart) != 64 {
		return false
	}
	_, err := hex.DecodeString(hexPart)
	return err == nil
}

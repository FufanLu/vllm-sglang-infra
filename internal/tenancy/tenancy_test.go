package tenancy

import (
	"errors"
	"strings"
	"testing"
)

func TestAuthenticate(t *testing.T) {
	d, err := New([]Tenant{{Name: "a"}, {Name: "b", Priority: Batch, Weight: 2}},
		map[string][]string{"a": {HashKey("ka1"), HashKey("ka2")}, "b": {strings.ToUpper(HashKey("kb"))}})
	if err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{"ka1": "a", "ka2": "a", "kb": "b"} {
		tn, err := d.Authenticate("Bearer " + key)
		if err != nil || tn.Name != want {
			t.Errorf("key %s -> %v, %v; want %s", key, tn, err, want)
		}
	}
	for _, h := range []string{"", "Bearer ", "Bearer nope", "Basic ka1", "ka1"} {
		if _, err := d.Authenticate(h); !errors.Is(err, ErrUnauthorized) {
			t.Errorf("header %q: err %v, want ErrUnauthorized", h, err)
		}
	}
	if tn, _ := d.Authenticate("Bearer ka1"); tn.Weight != 1 {
		t.Errorf("default weight = %v, want 1", tn.Weight)
	}
}

func TestNewRejectsBadHashes(t *testing.T) {
	if _, err := New([]Tenant{{Name: "a"}}, map[string][]string{"a": {"md5:abc"}}); err == nil {
		t.Error("malformed hash accepted")
	}
	h := HashKey("shared")
	if _, err := New([]Tenant{{Name: "a"}, {Name: "b"}}, map[string][]string{"a": {h}, "b": {h}}); err == nil {
		t.Error("one key for two tenants accepted")
	}
}

func TestOpenDirectory(t *testing.T) {
	tn, err := NewOpen().Authenticate("")
	if err != nil || tn != Anonymous {
		t.Errorf("open directory: %v, %v", tn, err)
	}
	if !strings.HasPrefix(HashKey("x"), "sha256:") || len(HashKey("x")) != 7+64 {
		t.Errorf("hash format: %s", HashKey("x"))
	}
}

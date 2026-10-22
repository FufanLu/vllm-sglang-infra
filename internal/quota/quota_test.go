package quota

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestMemoryBucket(t *testing.T) {
	now := time.Unix(0, 0)
	m := NewMemory()
	m.now = func() time.Time { return now }
	ctx := context.Background()

	if ok, _, _ := m.Reserve(ctx, "t", 600, 400); !ok {
		t.Fatal("full bucket refused")
	}
	ok, wait, _ := m.Reserve(ctx, "t", 600, 400) // 200 left, 200 missing at 10 tokens/s
	if ok || wait != 20*time.Second {
		t.Errorf("ok %v wait %v, want refused with 20s", ok, wait)
	}
	m.Adjust(ctx, "t", 600, -300) // actual usage was 100: refund 300
	if ok, _, _ := m.Reserve(ctx, "t", 600, 400); !ok {
		t.Error("refund not applied")
	}
	now = now.Add(time.Minute) // refills to the cap, not beyond
	if ok, _, _ := m.Reserve(ctx, "t", 600, 601); !ok {
		t.Error("a full bucket must admit even a request larger than the budget")
	}
	if ok, _, _ := m.Reserve(ctx, "t", 600, 1); ok {
		t.Error("bucket in debt admitted a request")
	}
	if ok, _, _ := m.Reserve(ctx, "unlimited", 0, 1<<40); !ok {
		t.Error("tenant without a budget refused")
	}
}

// TestRedis runs against a real Redis when VSI_TEST_REDIS_URL is set (CI does).
func TestRedis(t *testing.T) {
	url := os.Getenv("VSI_TEST_REDIS_URL")
	if url == "" {
		t.Skip("VSI_TEST_REDIS_URL not set")
	}
	r, err := NewRedis(url, "vsi:test:"+time.Now().Format("150405.000000")+":")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	ctx := context.Background()
	if ok, _, err := r.Reserve(ctx, "t", 600, 400); !ok || err != nil {
		t.Fatalf("full bucket: ok %v err %v", ok, err)
	}
	ok, wait, err := r.Reserve(ctx, "t", 600, 400)
	if ok || err != nil || wait < 19*time.Second || wait > 21*time.Second {
		t.Errorf("ok %v wait %v err %v, want refused with ~20s", ok, wait, err)
	}
	if err := r.Adjust(ctx, "t", 600, -300); err != nil {
		t.Fatal(err)
	}
	if ok, _, _ := r.Reserve(ctx, "t", 600, 400); !ok {
		t.Error("refund not applied")
	}
}

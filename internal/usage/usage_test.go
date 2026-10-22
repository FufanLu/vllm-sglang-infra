package usage

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"
)

var quiet = slog.New(slog.NewTextHandler(io.Discard, nil))

func TestJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	s, err := NewJSONL(path, quiet)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		s.Write(Record{Time: time.Unix(int64(i), 0).UTC(), Tenant: "a", PromptTokens: i, External: i == 2, CostUSD: 0.5})
	}
	if err := s.Close(); err != nil { // flushes
		t.Fatal(err)
	}
	f, _ := os.Open(path)
	defer f.Close()
	var got []Record
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r Record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatalf("line %q: %v", sc.Text(), err)
		}
		got = append(got, r)
	}
	if len(got) != 3 || got[2].PromptTokens != 2 || !got[2].External || got[0].Tenant != "a" {
		t.Errorf("records = %+v", got)
	}
}

// TestPostgres runs against a real database when VSI_TEST_POSTGRES_DSN is set (CI does).
func TestPostgres(t *testing.T) {
	dsn := os.Getenv("VSI_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("VSI_TEST_POSTGRES_DSN not set")
	}
	ctx := context.Background()
	p, err := NewPostgres(ctx, dsn, quiet)
	if err != nil {
		t.Fatal(err)
	}
	id := "test-" + time.Now().Format("150405.000000")
	p.Write(Record{Time: time.Now(), RequestID: id, Tenant: "pg", Model: "m", Backend: "b", Route: "chat",
		Status: "200", PromptTokens: 10, CompletionTokens: 5})
	p.Close()
	p2, err := NewPostgres(ctx, dsn, quiet)
	if err != nil {
		t.Fatal(err)
	}
	defer p2.Close()
	var n, prompt int
	if err := p2.pool.QueryRow(ctx, `SELECT count(*), coalesce(sum(prompt_tokens),0) FROM usage_records WHERE request_id=$1`, id).Scan(&n, &prompt); err != nil {
		t.Fatal(err)
	}
	if n != 1 || prompt != 10 {
		t.Errorf("rows %d, prompt tokens %d", n, prompt)
	}
}

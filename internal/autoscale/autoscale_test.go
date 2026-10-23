package autoscale

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"
	"testing"
	"time"
)

type fakeProvider struct {
	reps     []Replica
	launched int
	drained  []string
}

func (f *fakeProvider) List(context.Context) ([]Replica, error) { return slices.Clone(f.reps), nil }

func (f *fakeProvider) Launch(context.Context) (Replica, error) {
	f.launched++
	r := Replica{ID: fmt.Sprintf("new%d", f.launched), State: Pending}
	f.reps = append(f.reps, r)
	return r, nil
}

func (f *fakeProvider) Drain(_ context.Context, r Replica) error {
	f.drained = append(f.drained, r.ID)
	for i := range f.reps {
		if f.reps[i].ID == r.ID {
			f.reps[i].Draining = true
		}
	}
	return nil
}

func scaler(p Provider, load *int, now *time.Time) *Autoscaler {
	a := New(Config{Min: 1, Max: 4, TargetPerReplica: 10, ScaleDownAfter: 5 * time.Minute, HandoverLead: 20 * time.Minute},
		p, func() (int, int) { return *load, 0 }, slog.New(slog.NewTextHandler(io.Discard, nil)))
	a.now = func() time.Time { return *now }
	return a
}

func TestScaleUpImmediatelyWithinMax(t *testing.T) {
	now, load := time.Now(), 35
	p := &fakeProvider{reps: []Replica{{ID: "a", State: Running}}}
	d, err := scaler(p, &load, &now).Step(context.Background())
	if err != nil || d.Desired != 4 || d.Launched != 3 {
		t.Errorf("decision %+v err %v, want 3 launched for 35 requests at 10 per replica", d, err)
	}
	load = 500 // far above: still capped at Max
	if d, _ := scaler(p, &load, &now).Step(context.Background()); d.Launched != 0 {
		t.Errorf("launched beyond max: %+v", d)
	}
}

func TestScaleDownWaitsAndCancelsPendingFirst(t *testing.T) {
	now, load := time.Now(), 5
	p := &fakeProvider{reps: []Replica{
		{ID: "run1", State: Running, EndsAt: now.Add(6 * time.Hour)},
		{ID: "run2", State: Running, EndsAt: now.Add(3 * time.Hour)},
		{ID: "queued", State: Pending},
	}}
	a := scaler(p, &load, &now)
	if d, _ := a.Step(context.Background()); d.Drained != 0 {
		t.Fatalf("drained at once: %+v", d)
	}
	now = now.Add(6 * time.Minute)
	d, _ := a.Step(context.Background())
	if d.Drained != 2 || !slices.Equal(p.drained, []string{"queued", "run2"}) {
		t.Errorf("drained %v (%+v), want the queued job, then the replica ending soonest", p.drained, d)
	}
}

func TestLoadSpikeResetsScaleDownTimer(t *testing.T) {
	now, load := time.Now(), 5
	p := &fakeProvider{reps: []Replica{{ID: "a", State: Running}, {ID: "b", State: Running}}}
	a := scaler(p, &load, &now)
	a.Step(context.Background())
	now = now.Add(4 * time.Minute)
	load = 15
	a.Step(context.Background()) // demand back up: timer resets
	load = 5
	now = now.Add(2 * time.Minute)
	if d, _ := a.Step(context.Background()); d.Drained != 0 {
		t.Errorf("drained %d after only 2 minutes of low load", d.Drained)
	}
}

func TestHandoverBeforeWalltime(t *testing.T) {
	now, load := time.Now(), 10
	p := &fakeProvider{reps: []Replica{{ID: "old", State: Running, EndsAt: now.Add(15 * time.Minute)}}}
	d, _ := scaler(p, &load, &now).Step(context.Background())
	if d.Expiring != 1 || d.Launched != 1 || len(p.drained) != 0 {
		t.Errorf("decision %+v drained %v: a successor should be launched while the old replica keeps serving", d, p.drained)
	}
}

func TestSlurmProvider(t *testing.T) {
	var calls []string
	s := &Slurm{Script: "/c/vllm_replica.sbatch", Args: []string{"Qwen3-8B", "--max-model-len", "8192"}, JobName: "vsi-replica",
		Exec: func(_ context.Context, name string, args ...string) ([]byte, error) {
			calls = append(calls, name+" "+strings.Join(args, " "))
			switch name {
			case "squeue":
				return []byte("101|RUNNING|2026-10-22T06:30:00\n102|PENDING|N/A\n103|COMPLETING|2026-10-22T01:00:00\n"), nil
			case "sbatch":
				return []byte("104\n"), nil
			}
			return nil, nil
		}}
	ctx := context.Background()
	reps, err := s.List(ctx)
	if err != nil || len(reps) != 3 {
		t.Fatalf("list = %v, %v", reps, err)
	}
	if reps[0].State != Running || reps[0].EndsAt.Hour() != 6 || reps[1].State != Pending || !reps[1].EndsAt.IsZero() || !reps[2].Draining {
		t.Errorf("parsed %+v", reps)
	}
	r, _ := s.Launch(ctx)
	s.Drain(ctx, reps[0])
	s.Drain(ctx, reps[1])
	want := []string{
		"squeue -h --me -n vsi-replica -o %i|%T|%e",
		"sbatch --parsable --job-name vsi-replica /c/vllm_replica.sbatch Qwen3-8B --max-model-len 8192",
		"scancel --signal=TERM --batch 101",
		"scancel 102",
	}
	if r.ID != "104" || !slices.Equal(calls, want) {
		t.Errorf("launched %q, calls:\n%s", r.ID, strings.Join(calls, "\n"))
	}
	if reps, _ := s.List(ctx); !reps[0].Draining {
		t.Error("a drained job should stay marked while it is still in the queue")
	}
}

func TestSlurmLaunchFindsJobAcceptedDespiteError(t *testing.T) {
	sbatches := 0
	queue := "101|RUNNING|2026-10-22T06:30:00\n"
	s := &Slurm{Script: "/c/r.sbatch", JobName: "pool", RecheckWait: time.Millisecond,
		Exec: func(_ context.Context, name string, _ ...string) ([]byte, error) {
			switch name {
			case "squeue":
				return []byte(queue), nil
			case "sbatch":
				sbatches++
				// slurmctld queued the job, then failed the reply.
				queue += "102|PENDING|N/A\n"
				return nil, errors.New("sbatch: error: Batch job submission failed: Unexpected message received")
			}
			return nil, nil
		}}
	ctx := context.Background()
	s.List(ctx)
	r, err := s.Launch(ctx)
	if err != nil || r.ID != "102" || sbatches != 1 {
		t.Errorf("launch = %+v, %v after %d sbatch calls; want the accepted job 102 and no resubmission", r, err, sbatches)
	}
}

func TestSlurmHoldsSubmissionsAfterAFailedSbatch(t *testing.T) {
	sbatches := 0
	queue := "101|RUNNING|2026-10-22T06:30:00\n"
	clock := time.Date(2026, 10, 22, 22, 25, 0, 0, time.UTC)
	s := &Slurm{Script: "/c/r.sbatch", JobName: "pool", RecheckWait: time.Millisecond, Now: func() time.Time { return clock },
		Exec: func(_ context.Context, name string, _ ...string) ([]byte, error) {
			switch name {
			case "squeue":
				return []byte(queue), nil
			case "sbatch":
				sbatches++
				return nil, errors.New("sbatch: error: Batch job submission failed: Unexpected message received")
			}
			return nil, nil
		}}
	ctx := context.Background()
	s.List(ctx)
	if _, err := s.Launch(ctx); err == nil {
		t.Fatal("first launch: the job has not shown up, so it should fail")
	}
	clock = clock.Add(time.Minute)
	if _, err := s.Launch(ctx); err == nil || sbatches != 1 {
		t.Fatalf("within the hold: err %v after %d sbatch calls; want an error and no new submission", err, sbatches)
	}
	queue += "102|PENDING|N/A\n" // the failed submission's job appears late
	clock = clock.Add(30 * time.Second)
	if r, err := s.Launch(ctx); err != nil || r.ID != "102" || sbatches != 1 {
		t.Errorf("launch = %+v, %v after %d sbatch calls; want job 102 adopted", r, err, sbatches)
	}
}

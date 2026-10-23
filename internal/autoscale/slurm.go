package autoscale

import (
	"cmp"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// Slurm runs replicas as batch jobs of Script (cluster/vllm_replica.sbatch),
// which registers with the gateway once its model server is up and drains
// itself on SIGTERM.
type Slurm struct {
	Script  string
	Args    []string // the script's arguments: model name, vllm serve flags
	JobName string   // identifies this pool's jobs in squeue
	// Exec runs a Slurm command; tests replace it.
	Exec func(ctx context.Context, name string, args ...string) ([]byte, error)
	// RecheckWait is how long Launch waits before looking for a job that a
	// failed sbatch may have created anyway (default 5 s).
	RecheckWait time.Duration
	// SubmitHold is how long after a failed sbatch Launch only looks for
	// that job and submits nothing (default 3 min): slurmctld has queued
	// such a job 22 s after the error.
	SubmitHold time.Duration
	// Now is the clock; tests replace it.
	Now func() time.Time

	mu       sync.Mutex
	draining map[string]bool
	listed   map[string]bool // job IDs in the last List
	failedAt time.Time       // last failed sbatch
	unsure   map[string]bool // jobs listed before that sbatch
}

func (s *Slurm) run(ctx context.Context, name string, args ...string) ([]byte, error) {
	if s.Exec != nil {
		return s.Exec(ctx, name, args...)
	}
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if ee, ok := err.(*exec.ExitError); ok {
		return out, fmt.Errorf("%s: %w: %s", name, err, strings.TrimSpace(string(ee.Stderr)))
	}
	return out, err
}

// squeue's %e is the job's end time, like 2026-10-21T23:59:59, or N/A.
const slurmTime = "2006-01-02T15:04:05"

func (s *Slurm) List(ctx context.Context) ([]Replica, error) {
	out, err := s.run(ctx, "squeue", "-h", "--me", "-n", s.JobName, "-o", "%i|%T|%e")
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var reps []Replica
	seen := map[string]bool{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		f := strings.Split(strings.TrimSpace(line), "|")
		if len(f) != 3 {
			continue
		}
		r := Replica{ID: f[0], Draining: s.draining[f[0]]}
		switch f[1] {
		case "RUNNING":
			r.State = Running
		case "PENDING", "CONFIGURING":
			r.State = Pending
		default: // COMPLETING, CANCELLED, ...: on its way out
			r.Draining = true
		}
		if t, err := time.ParseInLocation(slurmTime, f[2], time.Local); err == nil {
			r.EndsAt = t
		}
		seen[r.ID] = true
		reps = append(reps, r)
	}
	for id := range s.draining {
		if !seen[id] {
			delete(s.draining, id) // gone from the queue
		}
	}
	s.listed = seen
	return reps, nil
}

func (s *Slurm) Launch(ctx context.Context) (Replica, error) {
	args := append([]string{"--parsable", "--job-name", s.JobName, s.Script}, s.Args...)
	now := s.Now
	if now == nil {
		now = time.Now
	}
	s.mu.Lock()
	before, failedAt, unsure := s.listed, s.failedAt, s.unsure
	s.mu.Unlock()
	// slurmctld under load can fail the reply and queue the job anyway
	// ("Unexpected message received", "Socket timed out"), even after the
	// error. Submitting again would start a second replica, so for a while
	// after a failure only look for that job.
	if !failedAt.IsZero() && now().Sub(failedAt) < cmp.Or(s.SubmitHold, 3*time.Minute) {
		if r, ok := s.findNew(ctx, unsure, 1); ok {
			s.resolve()
			return r, nil
		}
		return Replica{}, fmt.Errorf("sbatch failed at %s; waiting to see whether it queued a job", failedAt.Format(time.TimeOnly))
	}
	out, err := s.run(ctx, "sbatch", args...)
	if err != nil {
		s.mu.Lock()
		s.failedAt, s.unsure = now(), before
		s.mu.Unlock()
		if r, ok := s.findNew(ctx, before, 3); ok {
			s.resolve()
			return r, nil
		}
		return Replica{}, err
	}
	s.resolve()
	id, _, _ := strings.Cut(strings.TrimSpace(string(out)), ";") // "123;cluster" on federated setups
	return Replica{ID: id, State: Pending}, nil
}

func (s *Slurm) resolve() {
	s.mu.Lock()
	s.failedAt, s.unsure = time.Time{}, nil
	s.mu.Unlock()
}

// findNew lists the pool up to tries times and returns a job that was not
// in before.
func (s *Slurm) findNew(ctx context.Context, before map[string]bool, tries int) (Replica, bool) {
	wait := cmp.Or(s.RecheckWait, 5*time.Second)
	for range tries {
		select {
		case <-ctx.Done():
			return Replica{}, false
		case <-time.After(wait):
		}
		reps, err := s.List(ctx)
		if err != nil {
			continue
		}
		for _, r := range reps {
			if !before[r.ID] && !r.Draining {
				return r, true
			}
		}
	}
	return Replica{}, false
}

// Drain cancels a queued job outright; a running one gets SIGTERM in its
// batch shell, which the replica script turns into deregister-and-drain.
func (s *Slurm) Drain(ctx context.Context, r Replica) error {
	args := []string{r.ID}
	if r.State == Running {
		args = []string{"--signal=TERM", "--batch", r.ID}
	}
	if _, err := s.run(ctx, "scancel", args...); err != nil {
		return err
	}
	s.mu.Lock()
	if s.draining == nil {
		s.draining = map[string]bool{}
	}
	s.draining[r.ID] = true
	s.mu.Unlock()
	return nil
}

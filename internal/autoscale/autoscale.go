// Package autoscale sizes the replica pool from demand and keeps it alive
// across Slurm walltimes.
//
// Each pass compares demand — requests in flight plus requests queued in the
// gateway — with the capacity of the replicas that will still be around:
//
//   - Scale up at once when demand exceeds capacity: the cost of waiting is
//     queueing, and a new replica takes minutes to load.
//   - Scale down only after demand has stayed low for ScaleDownAfter, and
//     cancel replicas that are still queued in Slurm before draining ones
//     that run, since queued ones have no warm cache to lose.
//   - Replace replicas before their walltime: one that ends within
//     HandoverLead no longer counts as capacity, so a successor is submitted
//     while it still serves. The old replica drains itself on Slurm's
//     SIGTERM.
package autoscale

import (
	"context"
	"log/slog"
	"math"
	"slices"
	"time"
)

type State int

const (
	Pending State = iota // submitted, waiting for resources or loading
	Running
)

type Replica struct {
	ID       string // Slurm job id, process id, ...
	State    State
	EndsAt   time.Time // walltime end; zero if none
	Draining bool
}

// Provider starts and stops replicas.
type Provider interface {
	List(ctx context.Context) ([]Replica, error)
	Launch(ctx context.Context) (Replica, error)
	// Drain stops a replica gracefully: it deregisters, finishes in-flight
	// requests, and exits.
	Drain(ctx context.Context, r Replica) error
}

// Demand reports the load signal: requests in flight and queued.
type Demand func() (inflight, queued int)

type Config struct {
	Min, Max         int
	TargetPerReplica int // in-flight requests one replica should carry
	ScaleDownAfter   time.Duration
	HandoverLead     time.Duration
	Interval         time.Duration
}

type Autoscaler struct {
	cfg      Config
	provider Provider
	demand   Demand
	log      *slog.Logger
	now      func() time.Time
	lowSince time.Time
}

func New(cfg Config, p Provider, d Demand, log *slog.Logger) *Autoscaler {
	return &Autoscaler{cfg: cfg, provider: p, demand: d, log: log, now: time.Now}
}

// Decision is what one pass did.
type Decision struct {
	Desired, Capacity int
	Launched, Drained int
	Expiring          int
}

// Step runs one reconcile pass.
func (a *Autoscaler) Step(ctx context.Context) (Decision, error) {
	reps, err := a.provider.List(ctx)
	if err != nil {
		return Decision{}, err
	}
	now := a.now()
	var active []Replica
	d := Decision{}
	for _, r := range reps {
		if r.Draining {
			continue
		}
		if r.State == Running && !r.EndsAt.IsZero() && r.EndsAt.Sub(now) < a.cfg.HandoverLead {
			d.Expiring++ // still serving, but not for long: not capacity
			continue
		}
		active = append(active, r)
	}
	d.Capacity = len(active)

	inflight, queued := a.demand()
	load := inflight + queued
	d.Desired = int(math.Ceil(float64(load) / float64(max(a.cfg.TargetPerReplica, 1))))
	d.Desired = min(max(d.Desired, a.cfg.Min), a.cfg.Max)

	switch {
	case d.Capacity < d.Desired:
		a.lowSince = time.Time{}
		for range d.Desired - d.Capacity {
			r, err := a.provider.Launch(ctx)
			if err != nil {
				return d, err
			}
			d.Launched++
			a.log.Info("replica launched", "id", r.ID, "desired", d.Desired, "capacity", d.Capacity, "load", load)
		}
	case d.Capacity > d.Desired:
		if a.lowSince.IsZero() {
			a.lowSince = now
		}
		if now.Sub(a.lowSince) < a.cfg.ScaleDownAfter {
			break
		}
		// Cancel queued replicas first, then those closest to their walltime.
		slices.SortFunc(active, func(x, y Replica) int {
			if x.State != y.State {
				return int(x.State) - int(y.State) // Pending (0) first
			}
			return x.EndsAt.Compare(y.EndsAt)
		})
		for _, r := range active[:d.Capacity-d.Desired] {
			if err := a.provider.Drain(ctx, r); err != nil {
				return d, err
			}
			d.Drained++
			a.log.Info("replica drained", "id", r.ID, "desired", d.Desired, "capacity", d.Capacity, "load", load)
		}
		a.lowSince = time.Time{}
	default:
		a.lowSince = time.Time{}
	}
	return d, nil
}

// Run reconciles every Interval until ctx ends.
func (a *Autoscaler) Run(ctx context.Context) {
	t := time.NewTicker(a.cfg.Interval)
	defer t.Stop()
	for {
		if _, err := a.Step(ctx); err != nil {
			a.log.Warn("autoscale pass failed", "err", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

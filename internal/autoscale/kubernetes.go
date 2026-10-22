package autoscale

import (
	"context"
	"strconv"

	"github.com/FufanLu/vllm-sglang-infra/internal/k8s"
)

// Kubernetes scales a Deployment of model-server pods. Kubernetes chooses
// which pod to remove on scale-down; the pod's terminationGracePeriod and a
// preStop drain keep in-flight requests from being cut off, and the gateway
// stops routing to it as soon as it turns unready.
type Kubernetes struct {
	Client interface {
		k8s.PodLister
		Scale(ctx context.Context, deployment string) (int, error)
		SetScale(ctx context.Context, deployment string, replicas int) error
	}
	Deployment string
	Selector   string
}

func (k *Kubernetes) List(ctx context.Context) ([]Replica, error) {
	pods, err := k.Client.Pods(ctx, k.Selector)
	if err != nil {
		return nil, err
	}
	desired, err := k.Client.Scale(ctx, k.Deployment)
	if err != nil {
		return nil, err
	}
	var reps []Replica
	for _, p := range pods {
		if p.Deleting {
			continue
		}
		r := Replica{ID: p.Name, State: Pending}
		if p.Running && p.Ready {
			r.State = Running
		}
		reps = append(reps, r)
	}
	// Pods not created yet still count: the Deployment will make them.
	for i := len(reps); i < desired; i++ {
		reps = append(reps, Replica{ID: "pending-" + strconv.Itoa(i), State: Pending})
	}
	return reps[:min(len(reps), max(desired, 0))], nil
}

func (k *Kubernetes) Launch(ctx context.Context) (Replica, error) {
	n, err := k.Client.Scale(ctx, k.Deployment)
	if err != nil {
		return Replica{}, err
	}
	if err := k.Client.SetScale(ctx, k.Deployment, n+1); err != nil {
		return Replica{}, err
	}
	return Replica{ID: "scale-" + strconv.Itoa(n+1), State: Pending}, nil
}

func (k *Kubernetes) Drain(ctx context.Context, _ Replica) error {
	n, err := k.Client.Scale(ctx, k.Deployment)
	if err != nil || n == 0 {
		return err
	}
	return k.Client.SetScale(ctx, k.Deployment, n-1)
}

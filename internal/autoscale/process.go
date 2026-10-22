package autoscale

import (
	"context"
	"fmt"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Process runs replicas as local processes, for demos and tests without a
// cluster. Command is the argv; "{port}" is replaced by a free port. Started
// replicas are announced through Register, and Deregister is called before a
// drained one is stopped.
type Process struct {
	Command    []string
	Register   func(name, url string) error
	Deregister func(name string)
	// Inflight reports the gateway's in-flight requests to a replica, so a
	// drain can wait for them.
	Inflight func(name string) int64

	mu    sync.Mutex
	procs map[string]*proc
	next  int
}

type proc struct {
	cmd      *exec.Cmd
	draining bool
}

func (p *Process) List(context.Context) ([]Replica, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	var reps []Replica
	for id, pr := range p.procs {
		reps = append(reps, Replica{ID: id, State: Running, Draining: pr.draining})
	}
	return reps, nil
}

func (p *Process) Launch(ctx context.Context) (Replica, error) {
	port, err := freePort()
	if err != nil {
		return Replica{}, err
	}
	argv := make([]string, len(p.Command))
	for i, a := range p.Command {
		argv[i] = strings.ReplaceAll(a, "{port}", strconv.Itoa(port))
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	if err := cmd.Start(); err != nil {
		return Replica{}, err
	}
	p.mu.Lock()
	if p.procs == nil {
		p.procs = map[string]*proc{}
	}
	p.next++
	id := fmt.Sprintf("proc-%d", p.next)
	p.procs[id] = &proc{cmd: cmd}
	p.mu.Unlock()
	go cmd.Wait()
	if err := p.Register(id, fmt.Sprintf("http://127.0.0.1:%d", port)); err != nil {
		cmd.Process.Kill()
		return Replica{}, err
	}
	return Replica{ID: id, State: Running}, nil
}

func (p *Process) Drain(_ context.Context, r Replica) error {
	p.mu.Lock()
	pr := p.procs[r.ID]
	if pr == nil || pr.draining {
		p.mu.Unlock()
		return nil
	}
	pr.draining = true
	p.mu.Unlock()
	p.Deregister(r.ID)
	go func() {
		for deadline := time.Now().Add(2 * time.Minute); time.Now().Before(deadline) && p.Inflight(r.ID) > 0; {
			time.Sleep(100 * time.Millisecond)
		}
		pr.cmd.Process.Kill()
		p.mu.Lock()
		delete(p.procs, r.ID)
		p.mu.Unlock()
	}()
	return nil
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

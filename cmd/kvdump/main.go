// Command kvdump prints the KV-cache events a model server publishes, for
// checking that a replica's events reach the gateway and decode.
//
//	kvdump -endpoint tcp://d4052:5557
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/go-zeromq/zmq4"

	"github.com/FufanLu/vllm-sglang-infra/internal/kvevents"
)

func main() {
	endpoint := flag.String("endpoint", "tcp://127.0.0.1:5557", "publisher address")
	topic := flag.String("topic", "", "topic prefix to subscribe to")
	n := flag.Int("n", 0, "exit after this many messages (0: run until interrupted)")
	timeout := flag.Duration("timeout", 0, "exit after this long without a message (0: wait forever)")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	sub := zmq4.NewSub(ctx)
	defer sub.Close()
	if err := sub.Dial(*endpoint); err != nil {
		fmt.Fprintln(os.Stderr, "dial:", err)
		os.Exit(1)
	}
	sub.SetOption(zmq4.OptionSubscribe, *topic)
	for i := 0; *n == 0 || i < *n; i++ {
		type res struct {
			msg zmq4.Msg
			err error
		}
		ch := make(chan res, 1)
		go func() { m, err := sub.Recv(); ch <- res{m, err} }()
		var r res
		if *timeout > 0 {
			select {
			case r = <-ch:
			case <-time.After(*timeout):
				fmt.Fprintln(os.Stderr, "no message within", *timeout)
				os.Exit(2)
			}
		} else {
			r = <-ch
		}
		if r.err != nil {
			fmt.Fprintln(os.Stderr, "recv:", r.err)
			os.Exit(1)
		}
		f := r.msg.Frames
		fmt.Printf("message: %d frames, topic %q\n", len(f), string(f[0]))
		if len(f) < 3 {
			continue
		}
		b, err := kvevents.Decode(f[2])
		if err != nil {
			fmt.Printf("  decode error: %v (payload %d bytes, starts %x)\n", err, len(f[2]), f[2][:min(16, len(f[2]))])
			continue
		}
		for _, e := range b.Events {
			fmt.Printf("  %s blocks=%d parent=%q tokens=%d block_size=%d medium=%s\n",
				e.Type, len(e.BlockHashes), e.ParentHash, len(e.TokenIDs), e.BlockSize, e.Medium)
		}
	}
}

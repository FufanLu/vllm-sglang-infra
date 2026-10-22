// Command kvrelay fans KV-cache events from many publishers out to many
// subscribers. Publishers that dial out (llm-d-inference-sim) connect to its
// frontend; gateways subscribe to its backend. vLLM replicas bind their own
// publishers and need no relay.
//
//	kvrelay -frontend tcp://*:5557 -backend tcp://*:5558
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"

	"github.com/go-zeromq/zmq4"
)

func main() {
	front := flag.String("frontend", "tcp://*:5557", "address publishers connect to")
	back := flag.String("backend", "tcp://*:5558", "address subscribers connect to")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	sub := zmq4.NewSub(ctx)
	defer sub.Close()
	if err := sub.Listen(*front); err != nil {
		log.Fatalf("listen %s: %v", *front, err)
	}
	sub.SetOption(zmq4.OptionSubscribe, "")
	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	if err := pub.Listen(*back); err != nil {
		log.Fatalf("listen %s: %v", *back, err)
	}
	log.Printf("kvrelay: publishers -> %s, subscribers <- %s", *front, *back)
	for {
		msg, err := sub.Recv()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			log.Printf("recv: %v", err)
			continue
		}
		if err := pub.Send(msg); err != nil {
			log.Printf("send: %v", err)
		}
	}
}

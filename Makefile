COMPOSE := docker compose -f deploy/local/docker-compose.yaml

.PHONY: build test vet linux up down push

build:
	go build -o bin/ ./cmd/...

test:
	go test -race ./...

vet:
	go vet ./...

# Static Linux binaries for the Slurm cluster.
linux:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -o bin/linux-amd64/ ./cmd/...

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

# Copy binaries and cluster scripts to Explorer (see cluster/README.md).
push:
	cluster/push.sh

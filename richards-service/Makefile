.PHONY: build test race vet fmt docker run

build:
	go build ./...

test:
	go test ./... -count=1

race:
	go test -race ./... -count=1

vet:
	go vet ./...

fmt:
	gofmt -w cmd internal

docker:
	docker build -t richards-service .

run:
	go run ./cmd/server

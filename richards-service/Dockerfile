# syntax=docker/dockerfile:1
#
# Multi-stage build for the Richards unsaturated-seepage service.

# ---- build stage ----
FROM golang:1.22-alpine AS build
WORKDIR /src

# Cache dependencies independently of source changes.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# Static binary (CGO_ENABLED=0) for a scratch-free minimal runtime image.
ENV CGO_ENABLED=0 GOOS=linux
RUN go build -trimpath -ldflags="-s -w" -o /out/server ./cmd/server

# ---- runtime stage ----
FROM alpine:3.20
RUN adduser -D -u 10001 appuser
WORKDIR /app
COPY --from=build /out/server /app/server
USER appuser
EXPOSE 8080
ENV LISTEN_ADDR=":8080"
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
ENTRYPOINT ["/app/server"]

// Command server runs the Richards unsaturated-seepage HTTP service.
package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"richards-service/internal/api"
	"richards-service/internal/job"
)

func main() {
	addr := os.Getenv("LISTEN_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	mgr := job.NewManager()
	handler := api.NewServer(mgr)

	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("richards-seepage-service listening on %s", addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}

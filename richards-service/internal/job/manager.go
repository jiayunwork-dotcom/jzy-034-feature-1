package job

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
	"sync/atomic"
	"time"
)

// Stats holds service-wide counters for the monitoring endpoint.
type Stats struct {
	StartedUnixNano int64 `json:"started_unix_nano"`
	Submitted       int64 `json:"jobs_submitted"`
	Completed       int64 `json:"jobs_completed"`
	Failed          int64 `json:"jobs_failed"`
	InFlight        int64 `json:"jobs_in_flight"`
	StepsAccepted   int64 `json:"steps_accepted"`
}

// Uptime returns process age.
func (s *Stats) Uptime() time.Duration {
	return time.Duration(time.Now().UnixNano() - atomic.LoadInt64(&s.StartedUnixNano))
}

// Manager owns run accounting. The actual computation is stateless with
// respect to the manager (each job builds its own solver), so concurrent
// jobs cannot intermix profiles or water budgets; the mutex only guards the
// counters and the (optional) result registry.
type Manager struct {
	stats    Stats
	mu       sync.Mutex
	registry map[string]any
}

// NewManager creates a Manager.
func NewManager() *Manager {
	m := &Manager{registry: map[string]any{}}
	atomic.StoreInt64(&m.stats.StartedUnixNano, time.Now().UnixNano())
	return m
}

// NewID generates an opaque job id.
func NewID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// RunFull executes one complete run with accounting.
func (m *Manager) RunFull(req Request) (string, *Result, error) {
	id := NewID()
	atomic.AddInt64(&m.stats.Submitted, 1)
	atomic.AddInt64(&m.stats.InFlight, 1)
	defer atomic.AddInt64(&m.stats.InFlight, -1)

	res, err := Run(id, req)
	if err != nil {
		atomic.AddInt64(&m.stats.Failed, 1)
		return id, res, err
	}
	atomic.AddInt64(&m.stats.Completed, 1)
	atomic.AddInt64(&m.stats.StepsAccepted, int64(len(res.Steps)))
	m.mu.Lock()
	m.registry[id] = res
	m.mu.Unlock()
	return id, res, nil
}

// RunSingle executes one single-step run with accounting.
func (m *Manager) RunSingle(req StepRequest) (string, *StepResultResponse, error) {
	id := NewID()
	atomic.AddInt64(&m.stats.Submitted, 1)
	atomic.AddInt64(&m.stats.InFlight, 1)
	defer atomic.AddInt64(&m.stats.InFlight, -1)

	res, err := RunStep(id, req)
	if err != nil {
		atomic.AddInt64(&m.stats.Failed, 1)
		return id, nil, err
	}
	atomic.AddInt64(&m.stats.Completed, 1)
	atomic.AddInt64(&m.stats.StepsAccepted, 1)
	m.mu.Lock()
	m.registry[id] = res
	m.mu.Unlock()
	return id, res, nil
}

// StatsSnapshot returns a copy of the counters.
func (m *Manager) StatsSnapshot() Stats {
	return Stats{
		StartedUnixNano: atomic.LoadInt64(&m.stats.StartedUnixNano),
		Submitted:       atomic.LoadInt64(&m.stats.Submitted),
		Completed:       atomic.LoadInt64(&m.stats.Completed),
		Failed:          atomic.LoadInt64(&m.stats.Failed),
		InFlight:        atomic.LoadInt64(&m.stats.InFlight),
		StepsAccepted:   atomic.LoadInt64(&m.stats.StepsAccepted),
	}
}

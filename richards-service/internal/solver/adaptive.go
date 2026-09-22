package solver

import "math"

// AggregatedStep is the accepted result of advancing one reporting interval,
// possibly using several internal adaptive substeps.
type AggregatedStep struct {
	TimeBefore    float64 // [s]
	TimeAfter     float64 // [s]
	Dt            float64 // reporting interval [s]
	ThetaBefore   []float64
	ThetaAfter    []float64
	HBefore       []float64
	HAfter        []float64
	TopFlux       float64 // end-of-interval face flux [m/s]
	BottomFlux    float64
	CumTopFlux    float64 // cumulative depth [m]
	CumBottomFlux float64
	StorageBefore float64
	StorageAfter  float64
	// MassBalanceResidual is StorageChange - integrated net flux [m].
	MassBalanceResidual float64
	MassBalanceRelative float64
	Iterations          int // total Picard iterations over the substeps
	Substeps            int // number of internal substeps used
}

// AdaptiveConfig controls internal substep cutting during a reporting
// interval.
type AdaptiveConfig struct {
	MinSubstepFactor float64 // smallest substep = interval / (1 << maxCuts)
	MaxCuts          int
}

// DefaultAdaptiveConfig is the standard substepping policy.
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{MinSubstepFactor: 1.0 / 64.0, MaxCuts: 6}
}

// StepAdaptive advances exactly interval seconds, internally cutting the
// interval in half whenever a substep fails to converge, and restoring the
// step size as the front moves on. The reporting-level flux is the final
// substep's end flux; cumulative accounting is exact per substep.
//
// The single-step API uses Step with the caller's dt directly; this routine
// shares the identical Step machinery, constitutive functions and grid.
func (s *Solver) StepAdaptive(interval float64, cfg AdaptiveConfig) (*AggregatedStep, error) {
	if interval <= 0 {
		return nil, &Failure{Kind: FailNonConvergence, Message: "interval must be positive"}
	}
	hBefore := append([]float64(nil), s.H...)
	thBefore := append([]float64(nil), s.Theta...)
	storageBefore := s.Storage()
	timeBefore := s.Time
	cumTopBefore, cumBotBefore := s.CumTop, s.CumBot

	dt := interval
	totalIters := 0
	substeps := 0
	var last *StepResult
	advanced := 0.0

	for advanced < interval-1e-12 {
		if advanced+dt > interval {
			dt = interval - advanced
		}
		res, err := s.Step(dt)
		if err != nil {
			if f, ok := cuttable(err); ok && substepsCanShrink(dt, interval, cfg) {
				dt *= 0.5
				continue
			} else if ok {
				return nil, f
			}
			return nil, err
		}
		last = res
		advanced += dt
		substeps++
		totalIters += res.Iterations
		// grow back toward the reporting interval
		if dt*2 <= interval-(advanced-1e-12)+1e-12 {
			dt *= 2
		}
	}

	storageChange := s.Storage() - storageBefore
	netFlux := (s.CumTop - cumTopBefore) - (s.CumBot - cumBotBefore)
	imbalance := storageChange - netFlux
	rel := 0.0
	if d := math.Abs(storageChange); d > 1e-14 {
		rel = imbalance / d
	}
	return &AggregatedStep{
		TimeBefore:          timeBefore,
		TimeAfter:           s.Time,
		Dt:                  interval,
		ThetaBefore:         thBefore,
		ThetaAfter:          append([]float64(nil), s.Theta...),
		HBefore:             hBefore,
		HAfter:              append([]float64(nil), s.H...),
		TopFlux:             last.TopFlux,
		BottomFlux:          last.BottomFlux,
		CumTopFlux:          s.CumTop,
		CumBottomFlux:       s.CumBot,
		StorageBefore:       storageBefore,
		StorageAfter:        s.Storage(),
		MassBalanceResidual: imbalance,
		MassBalanceRelative: rel,
		Iterations:          totalIters,
		Substeps:            substeps,
	}, nil
}

func cuttable(err error) (*Failure, bool) {
	if f, ok := err.(*Failure); ok {
		return f, f.Kind == FailNonConvergence
	}
	return nil, false
}

func substepsCanShrink(dt, interval float64, cfg AdaptiveConfig) bool {
	return dt >= interval*cfg.MinSubstepFactor*(1+1e-9)
}

// MarchAdaptive advances nIntervals reporting intervals, each with adaptive
// internal substepping.
func (s *Solver) MarchAdaptive(interval float64, nIntervals int, cfg AdaptiveConfig) ([]AggregatedStep, error) {
	if nIntervals <= 0 {
		return nil, &Failure{Kind: FailNonConvergence, Message: "number of intervals must be positive"}
	}
	out := make([]AggregatedStep, 0, nIntervals)
	for k := 0; k < nIntervals; k++ {
		agg, err := s.StepAdaptive(interval, cfg)
		if err != nil {
			return out, err
		}
		out = append(out, *agg)
	}
	return out, nil
}

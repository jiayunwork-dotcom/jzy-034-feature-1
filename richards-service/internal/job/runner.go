package job

import (
	"richards-service/internal/solver"
)

// Run validates and executes a complete infiltration run, assembling the
// full time series. A compute failure (non-convergence / out-of-range
// water content) is returned as solver.Failure; validation errors are
// *ValidationError. Both are reported, never hidden or clipped.
func Run(jobID string, req Request) (*Result, error) {
	s, g, verr := buildSolver(req)
	if verr != nil {
		return nil, verr
	}
	if e := validateTime(req.Time.TotalTime, req.Time.StepSize); e != nil {
		return nil, e
	}
	nSteps, err := solver.StepsFromDuration(req.Time.TotalTime, req.Time.StepSize)
	if err != nil {
		return nil, &ValidationError{Code: CodeTimeInvalid,
			Field: "time", Message: err.Error()}
	}

	p := s.Params
	res := &Result{
		JobID:           jobID,
		Grid:            gridInfo(g),
		MLocked:         p.M(),
		InitialStorageM: s.Storage(),
		Steps:           make([]StepOutput, 0, nSteps),
	}

	steps, err := s.MarchAdaptive(req.Time.StepSize, nSteps, solver.DefaultAdaptiveConfig())
	for k := range steps {
		sr := &steps[k]
		res.Steps = append(res.Steps, StepOutput{
			Index:             k + 1,
			TimeS:             sr.TimeAfter,
			TopFlux:           sr.TopFlux,
			BottomFlux:        sr.BottomFlux,
			CumTopFluxM:       sr.CumTopFlux,
			CumBottomFluxM:    sr.CumBottomFlux,
			StorageM:          sr.StorageAfter,
			MassBalanceResidM: sr.MassBalanceResidual,
			MassBalanceRel:    sr.MassBalanceRelative,
			Iterations:        sr.Iterations,
			Substeps:          sr.Substeps,
			Layers:            snapshots(g, sr.HAfter, sr.ThetaAfter),
		})
	}
	if err != nil {
		// partial results are still on res; caller decides what to surface
		return res, err
	}

	_, _, _, cumTop, cumBot := s.State()
	res.FinalStorageM = s.Storage()
	res.CumTopFluxM = cumTop
	res.CumBottomFluxM = cumBot
	// Whole-run closure check: S_end - S_start - (Qtop - Qbot).
	res.TotalMassResidualM = res.FinalStorageM - res.InitialStorageM - (cumTop - cumBot)
	return res, nil
}

// RunStep validates and advances exactly one time step. It uses the very
// same buildSolver + solver.Step machinery as Run.
func RunStep(jobID string, req StepRequest) (*StepResultResponse, error) {
	full := Request{
		Column:   req.Column,
		Material: req.Material,
		Initial:  req.Initial,
		Boundary: req.Boundary,
		Time:     TimeSpec{TotalTime: req.StepSize, StepSize: req.StepSize},
		Options:  req.Options,
	}
	s, g, verr := buildSolver(full)
	if verr != nil {
		return nil, verr
	}
	if e := validateTime(req.StepSize, req.StepSize); e != nil {
		return nil, e
	}
	hBef, thBef, _, _, _ := s.State()
	sr, err := s.Step(req.StepSize)
	if err != nil {
		return nil, err
	}
	out := &StepResultResponse{
		JobID:   jobID,
		Grid:    gridInfo(g),
		MLocked: s.Params.M(),
		Before:  snapshots(g, hBef, thBef),
		After:   snapshots(g, sr.HAfter, sr.ThetaAfter),
		Step: StepOutput{
			Index:             1,
			TimeS:             sr.TimeAfter,
			TopFlux:           sr.TopFlux,
			BottomFlux:        sr.BottomFlux,
			CumTopFluxM:       sr.CumTopFlux,
			CumBottomFluxM:    sr.CumBottomFlux,
			StorageM:          sr.StorageAfter,
			MassBalanceResidM: sr.MassBalanceResidual,
			MassBalanceRel:    sr.MassBalanceRelative,
			Iterations:        sr.Iterations,
			Substeps:          1,
			Layers:            snapshots(g, sr.HAfter, sr.ThetaAfter),
		},
	}
	return out, nil
}

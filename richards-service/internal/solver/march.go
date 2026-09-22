package solver

import "math"

// March advances nSteps steps of length dt, returning every accepted step.
// It is a thin loop over Step, so single-step and full-run paths share the
// identical constitutive functions, grid and discretisation. On failure the
// returned results up to that point are provided together with the error.
func (s *Solver) March(dt float64, nSteps int) ([]StepResult, error) {
	if nSteps <= 0 {
		return nil, &Failure{Kind: FailNonConvergence, Message: "number of steps must be positive"}
	}
	results := make([]StepResult, 0, nSteps)
	for k := 0; k < nSteps; k++ {
		res, err := s.Step(dt)
		if err != nil {
			return results, err
		}
		results = append(results, *res)
	}
	return results, nil
}

// StepsFromDuration converts total duration and step size into a number of
// steps, requiring the division to be exact within a small tolerance.
func StepsFromDuration(totalTime, dt float64) (int, error) {
	if dt <= 0 {
		return 0, &Failure{Kind: FailNonConvergence, Message: "time step must be positive"}
	}
	if totalTime <= 0 {
		return 0, &Failure{Kind: FailNonConvergence, Message: "total time must be positive"}
	}
	nf := totalTime / dt
	n := int(math.Round(nf))
	if n < 1 || math.Abs(float64(n)*dt-totalTime) > 1e-9*math.Max(totalTime, dt) {
		return 0, &Failure{Kind: FailNonConvergence, Message: "total_time must be an integer multiple of time_step"}
	}
	return n, nil
}

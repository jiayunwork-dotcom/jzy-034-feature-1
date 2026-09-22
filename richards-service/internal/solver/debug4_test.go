package solver

import (
	"fmt"
	"testing"

	"richards-service/internal/constitutive"
)

// Debug: does fixed Step(30) converge through the barrier breakthrough?
func TestDebugFixedStepLayered(t *testing.T) {
	fine := constitutive.Params{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	coarse := constitutive.Params{Alpha: 20.0, N: 3.0, ThetaR: 0.03, ThetaS: 0.45, Ks: 1e-4}
	prof := &Profile{Segments: []Segment{
		{Thickness: 0.4, NZ: 24, Params: fine},
		{Thickness: 0.6, NZ: 36, Params: coarse},
	}}
	g, err := NewLayeredGrid([]float64{0.4, 0.6}, []int{24, 36})
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewLayeredSolverFromTheta(prof, g, TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(0.15, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	fails := 0
	for k := 0; k < 140; k++ {
		if _, err := s.Step(30); err != nil {
			fmt.Printf("step %d (t=%.0f): %v\n", k, s.Time, err)
			fails++
			if fails > 5 {
				break
			}
		}
	}
	fmt.Printf("done: t=%.0f fails=%d theta[24]=%.4f\n", s.Time, fails, s.Theta[24])
}

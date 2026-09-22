package solver

import (
	"fmt"
	"testing"

	"richards-service/internal/constitutive"
)

// Debug: print front trajectories for candidate layered parameter sets.
func TestDebugFrontTrajectory(t *testing.T) {
	fine := constitutive.Params{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	coarse := constitutive.Params{Alpha: 20.0, N: 3.0, ThetaR: 0.03, ThetaS: 0.45, Ks: 1e-4}
	const th0 = 0.15

	prof := &Profile{Segments: []Segment{
		{Thickness: 0.4, NZ: 24, Params: fine},
		{Thickness: 0.6, NZ: 36, Params: coarse},
	}}
	g, err := NewLayeredGrid([]float64{0.4, 0.6}, []int{24, 36})
	if err != nil {
		t.Fatal(err)
	}
	th := make([]float64, g.NZ)
	for i := range th {
		th[i] = th0
	}
	s, err := NewLayeredSolverFromTheta(prof, g, TopPondedHead, 0.02,
		BottomFreeDrainage, th, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	ref, err := NewSolverFromTheta(fine, NewGrid(60, 1.0), TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(th0, 60), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	frontOf := func(sv *Solver, theta []float64) float64 {
		depth := -1.0
		for i, x := range theta {
			var thS float64
			if sv.Profile != nil {
				thS = sv.Profile.Segments[sv.Grid.MaterialIndex(i)].Params.ThetaS
			} else {
				thS = sv.Params.ThetaS
			}
			if x > th0+0.3*(thS-th0) {
				depth = sv.Grid.Z[i]
			}
		}
		return depth
	}
	for k := 0; k < 150; k++ {
		aL, err := s.StepAdaptive(60, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("layered %d: %v", k, err)
		}
		aR, err := ref.StepAdaptive(60, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("ref %d: %v", k, err)
		}
		if k%5 == 0 || (frontOf(s, aL.ThetaAfter) > 0.3 && frontOf(s, aL.ThetaAfter) < 0.5) {
			fmt.Printf("t=%6.0f  frontL=%.4f  frontR=%.4f  thetaL[23]=%.4f thetaL[24]=%.4f hL[23]=%.4f hL[24]=%.4f\n",
				aL.TimeAfter, frontOf(s, aL.ThetaAfter), frontOf(ref, aR.ThetaAfter),
				aL.ThetaAfter[23], aL.ThetaAfter[24], aL.HAfter[23], aL.HAfter[24])
		}
	}
}

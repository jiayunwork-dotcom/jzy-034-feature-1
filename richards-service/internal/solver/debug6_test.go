package solver

import (
	"fmt"
	"testing"
)

// Debug: does the LEGACY uniform solver also stall on repeated Step(30)
// with this config (NZ=60, testMaterial, theta0=0.15, pond 0.02)?
func TestDebugLegacyUniformStep30(t *testing.T) {
	p := testMaterial()
	mk := func(nz int) *Solver {
		g := NewGrid(nz, 1.0)
		s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02,
			BottomFreeDrainage, repeat(0.15, nz), DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	for _, nz := range []int{50, 60} {
		s := mk(nz)
		for k := 0; k < 6; k++ {
			_, err := s.Step(30)
			fmt.Printf("uniform NZ=%d Step(30) #%d: err=%v\n", nz, k, err)
			if err != nil {
				break
			}
		}
	}
	// and with theta0=0.12
	s := func() *Solver {
		g := NewGrid(60, 1.0)
		s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02,
			BottomFreeDrainage, repeat(0.12, 60), DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		return s
	}()
	for k := 0; k < 6; k++ {
		_, err := s.Step(30)
		fmt.Printf("uniform NZ=60 th0=0.12 Step(30) #%d: err=%v\n", k, err)
		if err != nil {
			break
		}
	}
}

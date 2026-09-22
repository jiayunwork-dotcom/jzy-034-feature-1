package solver

import (
	"fmt"
	"testing"
)

// Debug: is the non-uniform Step(60) failure layering-related or just a
// hard step? Compare layered-degenerate vs homogeneous on the same grid,
// and try smaller steps.
func TestDebugNonUniformConvergence(t *testing.T) {
	p := fineSand()
	g, err := NewLayeredGrid([]float64{0.5, 0.5}, []int{25, 20})
	if err != nil {
		t.Fatal(err)
	}
	mk := func() *Solver {
		s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02,
			BottomFreeDrainage, repeat(0.12, g.NZ), DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	// homogeneous solver, same non-uniform grid, Step(60)
	s := mk()
	for k := 0; k < 4; k++ {
		_, err := s.Step(60)
		fmt.Printf("homogeneous Step(60) #%d: err=%v\n", k, err)
		if err != nil {
			break
		}
	}
	// homogeneous solver, Step(30)
	s2 := mk()
	for k := 0; k < 4; k++ {
		_, err := s2.Step(30)
		fmt.Printf("homogeneous Step(30) #%d: err=%v\n", k, err)
		if err != nil {
			break
		}
	}
	// uniform grid NZ=45, Step(60), same material/theta
	s3, err := NewSolverFromTheta(p, NewGrid(45, 1.0), TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(0.12, 45), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	for k := 0; k < 4; k++ {
		_, err := s3.Step(60)
		fmt.Printf("uniform45 Step(60) #%d: err=%v\n", k, err)
		if err != nil {
			break
		}
	}
}

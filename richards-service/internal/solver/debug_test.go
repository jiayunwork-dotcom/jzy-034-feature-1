package solver

import (
	"math"
	"testing"
)

// Debug: compare analytic Jacobian entries against finite differences of
// the residual on a non-uniform grid.
func TestDebugJacobianFD(t *testing.T) {
	p := fineSand()
	prof := &Profile{Segments: []Segment{
		{Thickness: 0.5, NZ: 25, Params: p},
		{Thickness: 0.5, NZ: 20, Params: p},
	}}
	g, err := NewLayeredGrid([]float64{0.5, 0.5}, []int{25, 20})
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewLayeredSolverFromTheta(prof, g, TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(0.12, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	// take one small step to develop a front
	if _, err := s.Step(5); err != nil {
		t.Fatalf("priming step: %v", err)
	}
	dt := 60.0
	thetaOld := append([]float64(nil), s.Theta...)
	h0 := append([]float64(nil), s.H...)

	// residual-only evaluation
	resAt := func(h []float64) []float64 {
		r, _, _, _, _ := s.residual(h, thetaOld, dt)
		return r
	}

	// numeric Jacobian of G_i = dz_i*R_i wrt h_j (tridiagonal entries)
	nz := g.NZ
	eps := 1e-7
	maxRel := 0.0
	var maxAt string
	r0 := resAt(h0)
	_ = r0
	for j := 0; j < nz; j++ {
		hp := append([]float64(nil), h0...)
		d := eps * math.Max(1, math.Abs(h0[j]))
		hp[j] += d
		rp := resAt(hp)
		hm := append([]float64(nil), h0...)
		hm[j] -= d
		rm := resAt(hm)
		for i := 0; i < nz; i++ {
			if i != j-1 && i != j && i != j+1 {
				continue
			}
			fd := (g.Dzs[i]*rp[i] - g.Dzs[i]*rm[i]) / (2 * d)
			_ = fd
			_ = i
		}
		_ = j
	}
	t.Logf("max rel diff %g at %s", maxRel, maxAt)

	// Now check that the Newton step from the analytic Jacobian is a
	// descent direction: compute residual norm before/after a full step
	// using the same assembly as Step (replicate minimal pieces).
	r, th, kCell, _, _ := s.residual(h0, thetaOld, dt)
	_ = th
	// assemble Jacobian exactly as Step does (non-uniform branch)
	lower := make([]float64, nz)
	diag := make([]float64, nz)
	upper := make([]float64, nz)
	rhs := make([]float64, nz)
	dkCell := make([]float64, nz)
	for i := 0; i < nz; i++ {
		dkCell[i] = s.paramsAt(i).DKDh(h0[i])
		if math.IsNaN(dkCell[i]) || math.IsInf(dkCell[i], 0) {
			dkCell[i] = 0
		}
	}
	for i := 0; i < nz; i++ {
		capacity := s.paramsAt(i).DThetaDH(h0[i])
		if capacity < 0 || math.IsNaN(capacity) {
			capacity = 0
		}
		diag[i] = g.Dzs[i] * capacity / dt
	}
	for f := 1; f < nz; f++ {
		ku, kd := kCell[f-1], kCell[f]
		lUp, lDown, lFace := g.faceGeometry(f)
		kf := weightedFaceK(ku, kd, lUp, lDown)
		factor := 1.0 + (h0[f-1]-h0[f])/lFace
		dqup := dWeightedFaceKDUp(ku, kd, lUp, lDown)*dkCell[f-1]*factor + kf/lFace
		dqdn := dWeightedFaceKDDown(ku, kd, lUp, lDown)*dkCell[f]*factor - kf/lFace
		diag[f-1] += dqup
		upper[f-1] += dqdn
		lower[f] += -dqup
		diag[f] += -dqdn
	}
	dz0 := g.Dzs[0]
	ksSurf := s.cellParams[0].Ks
	k0Harm := s.mean.mean(ksSurf, kCell[0])
	factor0 := s.PondedH - h0[0] + dz0/2.0
	dq0dn := s.mean.dMeanDDown(ksSurf, kCell[0])*dkCell[0]*(2.0/dz0)*factor0 - k0Harm*2.0/dz0
	diag[0] += -dq0dn
	diag[nz-1] += dkCell[nz-1]
	for i := 0; i < nz; i++ {
		rhs[i] = -g.Dzs[i] * r[i]
	}
	// finite-difference check of the assembled Jacobian
	for j := 0; j < nz; j++ {
		d := 1e-7 * math.Max(1, math.Abs(h0[j]))
		hp := append([]float64(nil), h0...)
		hp[j] += d
		rp, _, _, _, _ := s.residual(hp, thetaOld, dt)
		hm := append([]float64(nil), h0...)
		hm[j] -= d
		rm, _, _, _, _ := s.residual(hm, thetaOld, dt)
		for i := j - 1; i <= j+1 && i < nz; i++ {
			if i < 0 {
				continue
			}
			fd := (g.Dzs[i]*rp[i] - g.Dzs[i]*rm[i]) / (2 * d)
			var an float64
			switch {
			case i == j-1:
				an = upper[i]
			case i == j:
				an = diag[i]
			case i == j+1:
				an = lower[i]
			}
			den := math.Max(1e-30, math.Abs(an))
			rel := math.Abs(fd-an) / den
			if rel > maxRel && math.Abs(fd) > 1e-12 {
				maxRel = rel
				maxAt = string(rune('A'+i)) + "," + string(rune('A'+j))
				t.Logf("J[%d,%d]: analytic=%.10g fd=%.10g rel=%.3g", i, j, an, fd, rel)
			}
		}
	}
	t.Logf("max relative Jacobian discrepancy: %g", maxRel)
	if maxRel > 1e-4 {
		t.Fatalf("Jacobian mismatch: %g", maxRel)
	}
}

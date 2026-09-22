package solver

import (
	"fmt"
	"math"
	"testing"

	"richards-service/internal/constitutive"
)

// Debug: replicate Step's Newton loop with tracing at the failing state.
func TestDebugInterfaceNewton(t *testing.T) {
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
	if _, err := s.Step(30); err != nil {
		t.Fatalf("first step: %v", err)
	}
	dt := 30.0
	thetaOld := append([]float64(nil), s.Theta...)
	hIt := append([]float64(nil), s.H...)
	nz := g.NZ

	maxDh := math.Inf(1)
	for iter := 0; iter < 16; iter++ {
		r, th, kCell, _, _ := s.residual(hIt, thetaOld, dt)
		_ = th
		maxR := maxAbs(r)
		am := 0
		for i := range r {
			if math.Abs(r[i]) > math.Abs(r[am]) {
				am = i
			}
		}
		fmt.Printf("iter %2d maxR=%.4g @cell %d (seg%d) maxDh(prev)=%.3g  h23=%.4f h24=%.4f th23=%.4f th24=%.4f\n",
			iter, maxR, am, g.MaterialIndex(am), maxDh, hIt[23], hIt[24], th[23], th[24])

		lower := make([]float64, nz)
		diag := make([]float64, nz)
		upper := make([]float64, nz)
		rhs := make([]float64, nz)
		dkCell := make([]float64, nz)
		for i := 0; i < nz; i++ {
			dkCell[i] = s.paramsAt(i).DKDh(hIt[i])
			if math.IsNaN(dkCell[i]) || math.IsInf(dkCell[i], 0) {
				dkCell[i] = 0
			}
		}
		for i := 0; i < nz; i++ {
			capacity := s.paramsAt(i).DThetaDH(hIt[i])
			if capacity < 0 || math.IsNaN(capacity) {
				capacity = 0
			}
			diag[i] = g.Dzs[i] * capacity / dt
		}
		for f := 1; f < nz; f++ {
			ku, kd := kCell[f-1], kCell[f]
			lUp, lDown, lFace := g.faceGeometry(f)
			kf := weightedFaceK(ku, kd, lUp, lDown)
			factor := 1.0 + (hIt[f-1]-hIt[f])/lFace
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
		factor0 := s.PondedH - hIt[0] + dz0/2.0
		dq0dn := s.mean.dMeanDDown(ksSurf, kCell[0])*dkCell[0]*(2.0/dz0)*factor0 - k0Harm*2.0/dz0
		diag[0] += -dq0dn
		diag[nz-1] += dkCell[nz-1]
		for i := 0; i < nz; i++ {
			rhs[i] = -g.Dzs[i] * r[i]
		}
		for i := 0; i < nz; i++ {
			off := math.Abs(upper[i]) + math.Abs(lower[i])
			if off == 0 && diag[i] == 0 {
				diag[i] = g.Dzs[i] * capacityFloor / dt
			}
		}
		dh, ok := thomas(lower, diag, upper, rhs)
		if !ok {
			t.Fatal("singular")
		}
		maxDh = maxAbs(dh)
		tryRelax := 1.0
		if maxDh > maxNewtonHeadStep {
			tryRelax = math.Min(tryRelax, maxNewtonHeadStep/maxDh)
		}
		trial := make([]float64, nz)
		descent := false
		merit := s.meritOf(r)
		for attempt := 0; attempt < 40; attempt++ {
			for i := range dh {
				trial[i] = hIt[i] + tryRelax*dh[i]
			}
			rr, _, _, _, _ := s.residual(trial, thetaOld, dt)
			tm := s.meritOf(rr)
			if !hasNonFinite(rr) && tm < merit*(1.0-armijoC*tryRelax) {
				descent = true
				break
			}
			tryRelax *= 0.5
		}
		fmt.Printf("         maxDh=%.4g tryRelax=%.3g descent=%v\n", maxDh, tryRelax, descent)
		if !descent {
			fmt.Println("NO DESCENT — stop")
			break
		}
		for i := range dh {
			hIt[i] = trial[i]
		}
	}
}

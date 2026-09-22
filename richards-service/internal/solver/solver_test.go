package solver

import (
	"math"
	"testing"

	"richards-service/internal/constitutive"
)

// testMaterial is the preset sand-like material shared by solver tests.
func testMaterial() constitutive.Params {
	return constitutive.Params{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
}

func testGrid(nz int) Grid { return NewGrid(nz, 1.0) }

func uniformThetaSolver(t *testing.T, th0 float64, topKind string, pond float64,
	bottom string, opts Options) *Solver {
	t.Helper()
	p := testMaterial()
	g := testGrid(50)
	s, err := NewSolverFromTheta(p, g, topKind, pond, bottom,
		repeat(th0, g.NZ), opts)
	if err != nil {
		t.Fatalf("build solver: %v", err)
	}
	return s
}

func repeat(v float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// frontDepth returns the deepest layer whose water content exceeds
// theta0 + frac*(thetaS-theta0); -1 if none.
func frontDepth(theta []float64, th0, thS, frac float64, g Grid) float64 {
	thr := th0 + frac*(thS-th0)
	depth := -1.0
	for i, th := range theta {
		if th > thr {
			depth = g.Z[i]
		}
	}
	return depth
}

func pondingSolver(t *testing.T, pond float64, opts Options) *Solver {
	t.Helper()
	return uniformThetaSolver(t, 0.15, TopPondedHead, pond, BottomFreeDrainage, opts)
}

func TestSingleStepMassClosure(t *testing.T) {
	s := pondingSolver(t, 0.02, DefaultOptions())
	res, err := s.Step(30)
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	// Storage change must equal net boundary flux depth exactly.
	want := res.StorageAfter - res.StorageBefore
	got := (res.TopFlux - res.BottomFlux) * res.Dt
	if math.Abs(want-got) > 1e-10 {
		t.Fatalf("single-step closure: dS=%v netFlux=%v diff=%v", want, got, want-got)
	}
	if math.Abs(res.MassBalanceResidual) > 1e-10 {
		t.Fatalf("reported closure residual %v", res.MassBalanceResidual)
	}
}

func TestAdaptiveMarchMassClosureEveryInterval(t *testing.T) {
	s := pondingSolver(t, 0.02, DefaultOptions())
	steps, err := s.MarchAdaptive(30, 180, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	for k := range steps {
		if math.Abs(steps[k].MassBalanceResidual) > 1e-9 {
			t.Fatalf("interval %d closure residual %v", k, steps[k].MassBalanceResidual)
		}
	}
	// Global closure over the whole run.
	tot := s.Storage() - steps[0].StorageBefore - (s.CumTop - s.CumBot)
	if math.Abs(tot) > 1e-9 {
		t.Fatalf("global closure residual %v", tot)
	}
}

func TestWettingFrontMovesDownAndStorageRises(t *testing.T) {
	s := pondingSolver(t, 0.02, DefaultOptions())
	steps, err := s.MarchAdaptive(30, 180, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	f1 := frontDepth(steps[50].ThetaAfter, 0.15, 0.40, 0.3, s.Grid)
	f2 := frontDepth(steps[179].ThetaAfter, 0.15, 0.40, 0.3, s.Grid)
	if f1 < 0 || f2 <= f1 {
		t.Fatalf("front did not advance: f(1530s)=%v f(5400s)=%v", f1, f2)
	}
	if steps[179].StorageAfter <= steps[0].StorageBefore {
		t.Fatal("storage did not rise")
	}
}

func TestLargerPondedHeadDeeperAndWetter(t *testing.T) {
	run := func(pond float64) (float64, float64) {
		s := pondingSolver(t, pond, DefaultOptions())
		steps, err := s.MarchAdaptive(30, 90, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("march pond=%g: %v", pond, err)
		}
		last := steps[len(steps)-1]
		return frontDepth(last.ThetaAfter, 0.15, 0.40, 0.3, s.Grid), last.StorageAfter
	}
	fLo, sLo := run(0.005)
	fHi, sHi := run(0.20)
	if !(fHi > fLo+0.01) {
		t.Fatalf("front depth: low pond %v vs high pond %v", fLo, fHi)
	}
	if !(sHi > sLo+0.005) {
		t.Fatalf("storage: low pond %v vs high pond %v", sLo, sHi)
	}
}

func TestLowerKsSlowsFront(t *testing.T) {
	run := func(ks float64) float64 {
		p := testMaterial()
		p.Ks = ks
		g := testGrid(50)
		s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02, BottomFreeDrainage,
			repeat(0.15, g.NZ), DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		steps, err := s.MarchAdaptive(30, 180, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("march ks=%g: %v", ks, err)
		}
		last := steps[len(steps)-1]
		return frontDepth(last.ThetaAfter, 0.15, 0.40, 0.3, g)
	}
	fast := run(5e-5)
	slow := run(5e-6)
	if fast < 0.3 {
		t.Fatalf("baseline front unexpectedly shallow: %v", fast)
	}
	if slow > fast*0.5 {
		t.Fatalf("Ks/10 front %v not clearly slower than %v", slow, fast)
	}
}

func TestHydrostaticZeroFluxIsMotionless(t *testing.T) {
	p := testMaterial()
	g := testGrid(40)
	heads := make([]float64, g.NZ)
	for i := range heads {
		heads[i] = g.Z[i] - 1.5 // water table well below the column
	}
	s, err := NewSolver(p, g, TopZeroFlux, 0, BottomZeroFlux, heads, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	before := append([]float64(nil), s.Theta...)
	steps, err := s.MarchAdaptive(300, 48, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	for i := range before {
		if math.Abs(s.Theta[i]-before[i]) > 1e-12 {
			t.Fatalf("layer %d drifted: %.12f -> %.12f", i, before[i], s.Theta[i])
		}
	}
	if math.Abs(s.CumTop)+math.Abs(s.CumBot) > 0 {
		t.Fatalf("zero-flux boundaries produced flux: %g %g", s.CumTop, s.CumBot)
	}
	if math.Abs(steps[len(steps)-1].MassBalanceResidual) > 1e-12 {
		t.Fatal("hydrostatic closure not zero")
	}
}

func TestWaterContentAlwaysBounded(t *testing.T) {
	p := testMaterial()
	g := testGrid(50)
	s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.05, BottomFreeDrainage,
		repeat(0.10, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	steps, err := s.MarchAdaptive(30, 240, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	for k, st := range steps {
		for i, th := range st.ThetaAfter {
			if th < p.ThetaR-1e-12 || th > p.ThetaS+1e-12 {
				t.Fatalf("interval %d layer %d theta=%g out of [%g,%g]",
					k, i, th, p.ThetaR, p.ThetaS)
			}
		}
	}
}

func TestNonConvergenceReportedNotClipped(t *testing.T) {
	opts := DefaultOptions()
	opts.MaxIterations = 2
	s := pondingSolver(t, 0.05, opts)
	_, err := s.Step(50000.0) // deliberately brutal step with tiny iteration cap
	if err == nil {
		t.Fatal("expected convergence failure, got success")
	}
	f, ok := err.(*Failure)
	if !ok || f.Kind != FailNonConvergence {
		t.Fatalf("expected NON_CONVERGENCE failure, got %v", err)
	}
}

func TestOutOfRangeInitialRejected(t *testing.T) {
	p := testMaterial()
	g := testGrid(10)
	_, err := NewSolverFromTheta(p, g, TopZeroFlux, 0, BottomZeroFlux,
		repeat(p.ThetaS+0.01, g.NZ), DefaultOptions())
	if err == nil {
		t.Fatal("expected out-of-range initial content error")
	}
}

func TestHarmonicVsArithmeticMeanDistinguishable(t *testing.T) {
	build := func(m interblockMean) []float64 {
		s := pondingSolver(t, 0.05, DefaultOptions())
		s.mean = m
		steps, err := s.MarchAdaptive(30, 120, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("march mean=%v: %v", m, err)
		}
		return append([]float64(nil), steps[len(steps)-1].ThetaAfter...)
	}
	harm := build(HarmonicMean)
	arith := build(ArithmeticMean)
	maxDiff := 0.0
	for i := range harm {
		if d := math.Abs(harm[i] - arith[i]); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff < 1e-3 {
		t.Fatalf("harmonic and arithmetic interblock K indistinguishable, max diff %v", maxDiff)
	}
	// Arithmetic mean overstates conductivity across the orders-of-magnitude
	// jump at the front, so it must predict a deeper wetting front.
	g := testGrid(50)
	fh := frontDepth(harm, 0.15, 0.40, 0.3, g)
	fa := frontDepth(arith, 0.15, 0.40, 0.3, g)
	if !(fa > fh) {
		t.Fatalf("arithmetic front %v should exceed harmonic front %v", fa, fh)
	}
}

func TestHarmonicMeanDirectlySmallerThanArithmetic(t *testing.T) {
	k1, k2 := 1e-8, 5e-5
	h := HarmonicMean.mean(k1, k2)
	a := ArithmeticMean.mean(k1, k2)
	if !(h < a) {
		t.Fatalf("harmonic %v should be strictly below arithmetic %v", h, a)
	}
	if got := HarmonicMean.mean(0, 1); got != 0 {
		t.Fatalf("harmonic mean with zero K should be 0, got %v", got)
	}
}

func TestSingleStepAndAdaptiveIntervalSameResult(t *testing.T) {
	// When no internal substep is needed, StepAdaptive over an interval must
	// give the bit-identical result of one Step of the same length.
	direct := pondingSolver(t, 0.02, DefaultOptions())
	r1, err := direct.Step(30)
	if err != nil {
		t.Fatal(err)
	}
	adapt := pondingSolver(t, 0.02, DefaultOptions())
	agg, err := adapt.StepAdaptive(30, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatal(err)
	}
	if agg.Substeps != 1 {
		t.Fatalf("expected single substep, got %d", agg.Substeps)
	}
	for i := range r1.ThetaAfter {
		if math.Abs(r1.ThetaAfter[i]-agg.ThetaAfter[i]) > 1e-13 {
			t.Fatalf("layer %d differs: direct=%v adaptive=%v",
				i, r1.ThetaAfter[i], agg.ThetaAfter[i])
		}
	}
}

func TestAdaptiveCuttingConvergesWhereFixedStepStalls(t *testing.T) {
	// A reporting interval too long for one Newton solve at a sharp front is
	// recovered by internal halving; storage accounting still closes.
	s := pondingSolver(t, 0.05, DefaultOptions())
	agg, err := s.StepAdaptive(200, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("adaptive step: %v", err)
	}
	if agg.Substeps <= 1 {
		t.Fatalf("expected internal substepping, got %d", agg.Substeps)
	}
	if math.Abs(agg.MassBalanceResidual) > 1e-10 {
		t.Fatalf("adaptive closure residual %v", agg.MassBalanceResidual)
	}
}

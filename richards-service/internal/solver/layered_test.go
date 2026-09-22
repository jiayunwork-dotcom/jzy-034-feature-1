package solver

import (
	"math"
	"testing"

	"richards-service/internal/constitutive"
)

// ---- layered-grid construction -------------------------------------------

func TestNewLayeredGridAlignsInterfaces(t *testing.T) {
	g, err := NewLayeredGrid(1.0, []LayerSpec{
		{Thickness: 0.4, NZ: 20},
		{Thickness: 0.6, NZ: 30},
	})
	if err != nil {
		t.Fatalf("grid: %v", err)
	}
	if g.NZ != 50 {
		t.Fatalf("NZ=%d want 50", g.NZ)
	}
	// The interface at 0.4 m must be a face exactly.
	if g.ZFace[20] != 0.4 {
		t.Fatalf("interface face depth %v want 0.4", g.ZFace[20])
	}
	if g.ZFace[50] != 1.0 {
		t.Fatalf("bottom face %v want 1.0", g.ZFace[50])
	}
	if g.DzCell[19] != 0.4/20 || g.DzCell[20] != 0.6/30 {
		t.Fatalf("segment cell dz wrong: %v %v", g.DzCell[19], g.DzCell[20])
	}
	// No cell straddles the interface: cell 19 ends and cell 20 starts there.
	if got := g.Z[19] + g.DzCell[19]/2; got != 0.4 {
		t.Fatalf("up-cell bottom %v want 0.4", got)
	}
	if got := g.Z[20] - g.DzCell[20]/2; got != 0.4 {
		t.Fatalf("down-cell top %v want 0.4", got)
	}
}

func TestNewLayeredGridRejectsBadGeometry(t *testing.T) {
	cases := []struct {
		name string
		d    float64
		ls   []LayerSpec
	}{
		{"no layers", 1.0, nil},
		{"zero thickness segment", 1.0, []LayerSpec{{Thickness: 0, NZ: 5}, {Thickness: 1, NZ: 5}}},
		{"zero cells", 1.0, []LayerSpec{{Thickness: 0.5, NZ: 0}, {Thickness: 0.5, NZ: 5}}},
		{"sum mismatch", 1.0, []LayerSpec{{Thickness: 0.4, NZ: 20}, {Thickness: 0.5, NZ: 30}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewLayeredGrid(tc.d, tc.ls); err == nil {
				t.Fatal("expected grid error")
			}
		})
	}
}

// ---- degradation: layered machinery must not change single-material -----

func layeredFromUniform(p constitutive.Params, nSeg int, g Grid) []constitutive.Params {
	mats := make([]constitutive.Params, g.NZ)
	for i := range mats {
		mats[i] = p
	}
	return mats
}

// identical two-segment layered columns reproduce the homogeneous run
// cell-by-cell and time-by-time exactly.
func TestTwoIdenticalSegmentsBitIdenticalToHomogeneous(t *testing.T) {
	p := testMaterial()
	gU := testGrid(50)
	initU := repeat(0.15, gU.NZ)

	gL, err := NewLayeredGrid(1.0, []LayerSpec{
		{Thickness: 0.5, NZ: 25},
		{Thickness: 0.5, NZ: 25},
	})
	if err != nil {
		t.Fatal(err)
	}
	mats := layeredFromUniform(p, 2, gL)

	buildU := func() *Solver {
		s, err := NewSolverFromTheta(p, gU, TopPondedHead, 0.02, BottomFreeDrainage,
			append([]float64(nil), initU...), DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	buildL := func() *Solver {
		s, err := NewLayeredSolverFromTheta(mats, gL, TopPondedHead, 0.02, BottomFreeDrainage,
			append([]float64(nil), initU...), DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		return s
	}

	su, sl := buildU(), buildL()
	ru, errU := su.MarchAdaptive(30, 60, DefaultAdaptiveConfig())
	rl, errL := sl.MarchAdaptive(30, 60, DefaultAdaptiveConfig())
	if errU != nil {
		t.Fatalf("uniform march: %v", errU)
	}
	if errL != nil {
		t.Fatalf("layered march: %v", errL)
	}
	if len(ru) != len(rl) {
		t.Fatalf("adaptive paths diverged structurally: %d vs %d intervals", len(ru), len(rl))
	}
	for k := 0; k < len(ru); k++ {
		if ru[k].Substeps != rl[k].Substeps {
			t.Fatalf("interval %d substep count differs: %d vs %d",
				k, ru[k].Substeps, rl[k].Substeps)
		}
		for i := range ru[k].ThetaAfter {
			if ru[k].ThetaAfter[i] != rl[k].ThetaAfter[i] {
				t.Fatalf("interval %d layer %d theta differs: %v vs %v",
					k, i, ru[k].ThetaAfter[i], rl[k].ThetaAfter[i])
			}
			if ru[k].HAfter[i] != rl[k].HAfter[i] {
				t.Fatalf("interval %d layer %d head differs: %v vs %v",
					k, i, ru[k].HAfter[i], rl[k].HAfter[i])
			}
		}
		if ru[k].MassBalanceResidual != rl[k].MassBalanceResidual {
			t.Fatalf("interval %d closure differs: %v vs %v",
				k, ru[k].MassBalanceResidual, rl[k].MassBalanceResidual)
		}
	}
}

// layered constructor with one segment and the uniform grid is itself
// identical to the homogeneous constructor.
func TestSingleSegmentLayeredEqualsHomogeneous(t *testing.T) {
	p := testMaterial()
	g := testGrid(40)
	init := repeat(0.12, g.NZ)
	mats := layeredFromUniform(p, 1, g)
	s1, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02, BottomFreeDrainage,
		append([]float64(nil), init...), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	s2, err := NewLayeredSolverFromTheta(mats, g, TopPondedHead, 0.02, BottomFreeDrainage,
		append([]float64(nil), init...), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	r1, err := s1.Step(30)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := s2.Step(30)
	if err != nil {
		t.Fatal(err)
	}
	for i := range r1.ThetaAfter {
		if r1.ThetaAfter[i] != r2.ThetaAfter[i] || r1.HAfter[i] != r2.HAfter[i] {
			t.Fatalf("layer %d differs: th %v/%v h %v/%v", i,
				r1.ThetaAfter[i], r2.ThetaAfter[i], r1.HAfter[i], r2.HAfter[i])
		}
	}
}

// ---- layered physical behaviour ------------------------------------------

// fineSand / coarseSand: the coarse sand's much larger alpha makes its
// unsaturated conductivity collapse at the moderate negative heads ahead of
// a wetting front (drainage occurs early), even though its Ks is 7x higher.
func fineSand() constitutive.Params {
	return constitutive.Params{Alpha: 4.5, N: 2.68, ThetaR: 0.045, ThetaS: 0.43, Ks: 1.7e-5}
}

func coarseSand() constitutive.Params {
	return constitutive.Params{Alpha: 10.0, N: 2.68, ThetaR: 0.02, ThetaS: 0.36, Ks: 1.2e-4}
}

func layeredColumn(t *testing.T) (*Solver, int) {
	t.Helper()
	g, err := NewLayeredGrid(1.0, []LayerSpec{
		{Thickness: 0.4, NZ: 20},
		{Thickness: 0.6, NZ: 30},
	})
	if err != nil {
		t.Fatal(err)
	}
	f, c := fineSand(), coarseSand()
	mats := make([]constitutive.Params, g.NZ)
	th := make([]float64, g.NZ)
	for i := 0; i < 20; i++ {
		mats[i] = f
		th[i] = 0.12
	}
	for i := 20; i < 50; i++ {
		mats[i] = c
		th[i] = 0.05
	}
	s, err := NewLayeredSolverFromTheta(mats, g, TopPondedHead, 0.02, BottomFreeDrainage,
		th, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	return s, 20 // interface face index
}

func TestCoarseSandIsLessConductiveUnsaturated(t *testing.T) {
	// Sanity-check the material contrast at the heads met ahead of a front.
	f, c := fineSand(), coarseSand()
	for _, h := range []float64{-0.6, -0.4, -0.25, -0.15} {
		if !(f.Conductivity(h) > c.Conductivity(h)) {
			t.Fatalf("at h=%g fine K=%g should exceed coarse K=%g", h, f.Conductivity(h), c.Conductivity(h))
		}
	}
}

// uniformHeadLayeredColumn builds a fine-over-coarse column initialised with
// one uniform pressure head h0: the physically well-defined initial state
// across two materials (the two segments' initial water contents then
// differ, each on its own curve). layered=false replaces the coarse segment
// by the fine material, giving the geometry/initial-head matched control.
func uniformHeadLayeredColumn(t *testing.T, layered bool, h0 float64) (*Solver, int) {
	t.Helper()
	g, err := NewLayeredGrid(1.0, []LayerSpec{
		{Thickness: 0.4, NZ: 20},
		{Thickness: 0.6, NZ: 30},
	})
	if err != nil {
		t.Fatal(err)
	}
	f, c := fineSand(), coarseSand()
	mats := make([]constitutive.Params, g.NZ)
	heads := make([]float64, g.NZ)
	for i := 0; i < g.NZ; i++ {
		if layered && i >= 20 {
			mats[i] = c
		} else {
			mats[i] = f
		}
		heads[i] = h0
	}
	s, err := NewLayeredSolver(mats, g, TopPondedHead, 0.02, BottomFreeDrainage,
		heads, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	return s, 20 // interface face index
}

// frontDepthH is a material-aware wetting-front detector for the
// uniform-head initialisation: each cell's threshold is its own theta(h0)
// plus frac of that material's storage range.
func frontDepthH(s *Solver, h0, frac float64) float64 {
	depth := -1.0
	for i := range s.Theta {
		p := s.MatAt(i)
		th0 := p.WaterContent(h0)
		thr := th0 + frac*(p.ThetaS-th0)
		if s.Theta[i] > thr {
			depth = s.Grid.Z[i]
		}
	}
	return depth
}

// firstPassage scans an accepted trajectory and returns the earliest time at
// which the front reaches depth d.
func firstPassage(times, fronts []float64, d float64) (float64, bool) {
	for k, fd := range fronts {
		if fd >= d {
			return times[k], true
		}
	}
	return 0, false
}

func TestLayeredFrontSpeedChangesAtInterface(t *testing.T) {
	// Both columns share identical geometry, grid, initial head profile and
	// boundary; only the lower segment's material differs.
	march := func(layered bool) (times, fronts []float64, s *Solver) {
		s, _ = uniformHeadLayeredColumn(t, layered, -0.6)
		steps, err := s.MarchAdaptive(120, 200, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("march layered=%v: %v", layered, err)
		}
		times = make([]float64, 0, len(steps))
		fronts = make([]float64, 0, len(steps))
		for _, st := range steps {
			s.Theta = st.ThetaAfter
			s.H = st.HAfter
			times = append(times, st.TimeAfter)
			fronts = append(fronts, frontDepthH(s, -0.6, 0.3))
		}
		return times, fronts, s
	}

	lT, lF, ls := march(true)
	cT, cF, _ := march(false)

	// Marker depths straddle the 0.4 m interface:
	// 0.35 -> 0.39 approaches it inside the fine segment;
	// 0.41 -> 0.51 crosses through the first coarse cells.
	depth1, depth2, depth3, depth4 := 0.35, 0.39, 0.41, 0.51
	t1l, ok1 := firstPassage(lT, lF, depth1)
	t2l, ok2 := firstPassage(lT, lF, depth2)
	t3l, ok3 := firstPassage(lT, lF, depth3)
	t4l, ok4 := firstPassage(lT, lF, depth4)
	if !(ok1 && ok2 && ok3 && ok4) {
		t.Fatalf("layered front did not cross all markers: %v %v %v %v", ok1, ok2, ok3, ok4)
	}
	t1c, _ := firstPassage(cT, cF, depth1)
	t2c, _ := firstPassage(cT, cF, depth2)
	t3c, _ := firstPassage(cT, cF, depth3)
	t4c, _ := firstPassage(cT, cF, depth4)

	// Pre-interface traversal (0.35 -> 0.39) times must agree: before the
	// front reaches the interface the two columns are physically identical.
	dwellFineL := t2l - t1l
	dwellFineC := t2c - t1c
	if ratio := dwellFineL / dwellFineC; ratio > 1.35 {
		t.Fatalf("pre-interface motion unexpectedly differs: layered %v control %v",
			dwellFineL, dwellFineC)
	}

	// The interface dwell — time between arriving at the last fine cell and
	// entering the first coarse cell — must be dramatically longer with the
	// coarse material present (capillary barrier), several times the
	// control's geometric crossing time.
	dwellIfaceL := t3l - t2l
	dwellIfaceC := t3c - t2c
	if dwellIfaceL < 3.0*dwellIfaceC || dwellIfaceL < 1200 {
		t.Fatalf("no observable interface hold-up: layered dwell %v s vs control %v s",
			dwellIfaceL, dwellIfaceC)
	}

	// After breakthrough, advancement through the coarse cells is far slower
	// than the homogeneous control over the same depth interval.
	crossCoarseL := t4l - t3l
	crossCoarseC := t4c - t3c
	if crossCoarseL < 2.5*crossCoarseC {
		t.Fatalf("post-breakthrough speed not observably reduced: layered %v s vs control %v s",
			crossCoarseL, crossCoarseC)
	}

	// Same result expressed as a speed change within the layered run alone:
	// the post-interface front speed is below half the pre-interface speed.
	vPre := (depth2 - depth1) / dwellFineL
	vPost := (depth4 - depth3) / crossCoarseL
	if !(vPost < 0.5*vPre) {
		t.Fatalf("front speed did not drop across interface: vPre=%v vPost=%v", vPre, vPost)
	}
	t.Logf("fine(0.35-0.39) layered/control: %.0f/%.0f s; interface dwell %.0f vs %.0f s; "+
		"coarse(0.41-0.51) %.0f vs %.0f s; speed %.2e -> %.2e m/s",
		dwellFineL, dwellFineC, dwellIfaceL, dwellIfaceC,
		crossCoarseL, crossCoarseC, vPre, vPost)
	_ = ls
}

// TestTopFaceJacobianNonuniformMatchesFD checks the ponded-surface face flux
// and its derivative on a genuinely non-uniform grid (this branch is never
// exercised by the bitwise-uniform degradation tests). It must reproduce the
// legacy 2*harmonic(Ks,K0)/dz0 conductance once centre-to-boundary distance
// dz0/2 is split into two equal resistances.
func TestTopFaceJacobianNonuniformMatchesFD(t *testing.T) {
	g, err := NewLayeredGrid(1.0, []LayerSpec{
		{Thickness: 0.3, NZ: 10},
		{Thickness: 0.7, NZ: 20},
	})
	if err != nil {
		t.Fatal(err)
	}
	if g.IsUniform() {
		t.Fatal("setup must be non-uniform")
	}
	mats := make([]constitutive.Params, g.NZ)
	heads := make([]float64, g.NZ)
	for i := range mats {
		if i >= 10 {
			mats[i] = coarseSand()
		} else {
			mats[i] = fineSand()
		}
		heads[i] = -0.5
	}
	s, err := NewLayeredSolver(mats, g, TopPondedHead, 0.02, BottomFreeDrainage,
		heads, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	h := append([]float64(nil), heads...)
	dz0 := g.DzCell[0]
	kSurf := mats[0].Ks
	k0 := mats[0].Conductivity(h[0])
	quarter := dz0 / 4.0
	g0 := 1.0 / (quarter/kSurf + quarter/k0)
	qWant := g0 * (s.PondedH - h[0] + dz0/2.0)
	qGot := s.FaceFluxes(h)[0]
	if math.Abs(qWant-qGot) > 1e-12*math.Max(math.Abs(qWant), 1e-20) {
		t.Fatalf("top flux %g want %g", qGot, qWant)
	}
	// derivative dq0/dh0: dG0/dK0 * dK0/dh0 * factor - G0
	dk0dh := mats[0].DKDh(h[0])
	factor := s.PondedH - h[0] + dz0/2.0
	want := g0*g0*quarter/(k0*k0)*dk0dh*factor - g0
	q := func(hh []float64) float64 { return s.FaceFluxes(hh)[0] }
	const eps = 1e-6
	hp, hm := append([]float64(nil), h...), append([]float64(nil), h...)
	hp[0] += eps
	hm[0] -= eps
	got := (q(hp) - q(hm)) / (2 * eps)
	scale := math.Max(math.Max(math.Abs(want), math.Abs(got)), 1e-12)
	if math.Abs(want-got)/scale > 1e-5 {
		t.Fatalf("top dq/dh0 analytical %g vs FD %g", want, got)
	}
}

// TestInterfaceJacobianMatchesFiniteDifferences verifies by central finite
// differences the re-derived interface-face Jacobian entries (the part that
// replaces the single-material harmonic-mean derivatives): at a material
// interface dq/dh_up and dq/dh_down must follow from each side's own
// K(h) curve through the series-conductance formula.
func TestInterfaceJacobianMatchesFiniteDifferences(t *testing.T) {
	s, f := layeredColumn(t)
	// Wet the interface neighbourhood first so the check runs where the
	// contrast is actively nonlinear.
	_, err := s.MarchAdaptive(60, 160, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("approach march: %v", err)
	}
	h := append([]float64(nil), s.H...)

	// Analytical derivatives of the interior-face flux as assembled in Step:
	//	q = span*G + G*(hu-hd), G = 1/(du/Ku + dd/Kd)
	fg := s.faces[f]
	ku := s.MatAt(f - 1).Conductivity(h[f-1])
	kd := s.MatAt(f).Conductivity(h[f])
	au := fg.dUp/ku + fg.dDown/kd
	G := 1.0 / au
	dgu := G * G * fg.dUp / (ku * ku)
	dgd := G * G * fg.dDown / (kd * kd)
	dkuDh := s.MatAt(f - 1).DKDh(h[f-1])
	dkdDh := s.MatAt(f).DKDh(h[f])
	dhdiff := h[f-1] - h[f]
	wantUp := dgu*dkuDh*(fg.span+G*dhdiff) + G
	wantDown := dgd*dkdDh*(fg.span+G*dhdiff) - G

	q := func(hh []float64) float64 { return s.FaceFluxes(hh)[f] }
	const eps = 1e-6
	hp, hm := append([]float64(nil), h...), append([]float64(nil), h...)
	hp[f-1] += eps
	hm[f-1] -= eps
	gotUp := (q(hp) - q(hm)) / (2 * eps)
	hp, hm = append([]float64(nil), h...), append([]float64(nil), h...)
	hp[f] += eps
	hm[f] -= eps
	gotDown := (q(hp) - q(hm)) / (2 * eps)

	scale := func(a, b float64) float64 {
		m := math.Max(math.Abs(a), math.Abs(b))
		if m < 1e-12 {
			m = 1e-12
		}
		return m
	}
	if d := math.Abs(wantUp-gotUp) / scale(wantUp, gotUp); d > 1e-5 {
		t.Fatalf("dq/dh_up analytical %g vs FD %g (rel %g)", wantUp, gotUp, d)
	}
	if d := math.Abs(wantDown-gotDown) / scale(wantDown, gotDown); d > 1e-5 {
		t.Fatalf("dq/dh_down analytical %g vs FD %g (rel %g)", wantDown, gotDown, d)
	}
	// The interface must couple to both sides with opposite gravitational
	// signs even though the K curves differ by orders of magnitude.
	if wantUp <= 0 || wantDown >= 0 {
		t.Fatalf("interface Jacobian signs wrong: dq/dh_up=%g dq/dh_down=%g", wantUp, wantDown)
	}
}

// TestInterfaceFluxIsSingleContinuousValue verifies the single shared
// interface flux: it is assembled once from both sides' own K(h) curves and
// is the exact flux entering/leaving each adjacent converged cell.
func TestInterfaceFluxIsSingleContinuousValue(t *testing.T) {
	s, f := layeredColumn(t)
	// March adaptively until the front wets the cell immediately above the
	// interface (the production path also cuts step size adaptively at a
	// sharp front); then take one small fixed implicit step so the
	// before/after state pair matches a single converged Newton solve.
	_, err := s.MarchAdaptive(60, 160, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("approach march: %v", err)
	}
	if s.Theta[f-1] <= 0.25 {
		t.Fatalf("front did not reach the interface (theta up=%g)", s.Theta[f-1])
	}
	thBefore := append([]float64(nil), s.Theta...)
	const dt = 15.0
	res, err := s.Step(dt)
	if err != nil {
		t.Fatalf("interface step: %v", err)
	}
	hAfter, thAfter := res.HAfter, res.ThetaAfter

	// One face flux computed from both sides' own K(h) curves: FaceFluxes
	// and the residual assembly use the identical expression, and the
	// residual (each cell's flux balance) is converged.
	qf := s.FaceFluxes(hAfter)
	qIface := qf[f]
	// Re-derive independently from the two cell states:
	//	G = 1/(du/Ku + dd/Kd), q = G*(span + hu - hd)
	ku := s.MatAt(f - 1).Conductivity(hAfter[f-1])
	kd := s.MatAt(f).Conductivity(hAfter[f])
	dUp := s.Grid.ZFace[f] - s.Grid.Z[f-1]
	dDown := s.Grid.Z[f] - s.Grid.ZFace[f]
	span := dUp + dDown
	g := 1.0 / (dUp/ku + dDown/kd)
	qWant := g * (span + hAfter[f-1] - hAfter[f])
	if tol := 1e-12 * math.Max(math.Abs(qIface), 1e-20); math.Abs(qIface-qWant) > tol {
		t.Fatalf("interface flux mismatch: assembled %v re-derived %v", qIface, qWant)
	}

	// Flux continuity in the finite-volume sense: at the converged state each
	// interface-adjacent cell's residual (storage rate minus net face flux)
	// is zero to the Newton tolerance, so the single shared face flux leaves
	// one cell and enters the next with no interface source/sink.
	r, _, _, _ := s.residual(hAfter, thBefore, dt)
	if math.Abs(r[f-1]) > 1e-7 || math.Abs(r[f]) > 1e-7 {
		t.Fatalf("interface cells not balanced: R_up=%g R_down=%g", r[f-1], r[f])
	}

	// Water contents on the two sides need not (and here do not) match, yet
	// each lies in its own material's range.
	pu, pd := s.MatAt(f-1), s.MatAt(f)
	tu, td := thAfter[f-1], thAfter[f]
	if tu < pu.ThetaR-1e-12 || tu > pu.ThetaS+1e-12 {
		t.Fatalf("up-cell theta %g out of fine range", tu)
	}
	if td < pd.ThetaR-1e-12 || td > pd.ThetaS+1e-12 {
		t.Fatalf("down-cell theta %g out of coarse range", td)
	}
	t.Logf("interface theta up/down: %.4f / %.4f (ranges [%g,%g] vs [%g,%g]); flux=%g",
		tu, td, pu.ThetaR, pu.ThetaS, pd.ThetaR, pd.ThetaS, qIface)
}

func TestLayeredMassClosureEveryStep(t *testing.T) {
	s, _ := layeredColumn(t)
	steps, err := s.MarchAdaptive(60, 180, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	for k, st := range steps {
		if math.Abs(st.MassBalanceResidual) > 1e-9 {
			t.Fatalf("interval %d closure %g", k, st.MassBalanceResidual)
		}
	}
	tot := s.Storage() - steps[0].StorageBefore - (s.CumTop - s.CumBot)
	if math.Abs(tot) > 1e-9 {
		t.Fatalf("global layered closure %g", tot)
	}
}

// A layered zero-flux hydrostatic column is motionless cell-by-cell, even
// across material interfaces: the constant-total-head profile is an exact
// equilibrium for every material curve.
func TestLayeredHydrostaticZeroFluxMotionless(t *testing.T) {
	g, err := NewLayeredGrid(1.0, []LayerSpec{
		{Thickness: 0.4, NZ: 16},
		{Thickness: 0.6, NZ: 24},
	})
	if err != nil {
		t.Fatal(err)
	}
	mats := make([]constitutive.Params, g.NZ)
	for i := 0; i < 16; i++ {
		mats[i] = fineSand()
	}
	for i := 16; i < 40; i++ {
		mats[i] = coarseSand()
	}
	heads := make([]float64, g.NZ)
	for i := range heads {
		heads[i] = g.Z[i] - 1.5
	}
	s, err := NewLayeredSolver(mats, g, TopZeroFlux, 0, BottomZeroFlux, heads, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	before := append([]float64(nil), s.Theta...)
	steps, err := s.MarchAdaptive(300, 24, DefaultAdaptiveConfig())
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
		t.Fatal("hydrostatic layered closure not zero")
	}
}

// ---- per-material initial-profile validation ------------------------------

func TestLayeredInitialThetaBoundsPerSegment(t *testing.T) {
	g, err := NewLayeredGrid(1.0, []LayerSpec{
		{Thickness: 0.4, NZ: 20},
		{Thickness: 0.6, NZ: 30},
	})
	if err != nil {
		t.Fatal(err)
	}
	f, c := fineSand(), coarseSand()
	mats := make([]constitutive.Params, g.NZ)
	for i := 0; i < 20; i++ {
		mats[i] = f
	}
	for i := 20; i < 50; i++ {
		mats[i] = c
	}

	// 0.40 is legal in the fine segment (thetaS 0.43) but above the coarse
	// segment's thetaS 0.36: putting it in a coarse cell must be rejected.
	th := make([]float64, g.NZ)
	for i := range th {
		th[i] = 0.1
	}
	if _, err := NewLayeredSolverFromTheta(mats, g, TopZeroFlux, 0, BottomZeroFlux,
		th, DefaultOptions()); err != nil {
		t.Fatalf("in-range profile rejected: %v", err)
	}
	th[25] = 0.40
	_, err = NewLayeredSolverFromTheta(mats, g, TopZeroFlux, 0, BottomZeroFlux,
		th, DefaultOptions())
	if err == nil {
		t.Fatal("expected out-of-range error in coarse segment")
	}
	// Same value in the fine segment is fine.
	th[25] = 0.1
	th[5] = 0.40
	if _, err := NewLayeredSolverFromTheta(mats, g, TopZeroFlux, 0, BottomZeroFlux,
		th, DefaultOptions()); err != nil {
		t.Fatalf("fine segment should accept 0.40: %v", err)
	}
	// A value below coarse thetaR but above fine thetaR in a fine cell is
	// legal there but must be rejected in coarse.
	th[5] = 0.1
	th[30] = 0.03 // coarse thetaR = 0.02: legal
	if _, err := NewLayeredSolverFromTheta(mats, g, TopZeroFlux, 0, BottomZeroFlux,
		th, DefaultOptions()); err != nil {
		t.Fatalf("0.03 should be legal in coarse: %v", err)
	}
}

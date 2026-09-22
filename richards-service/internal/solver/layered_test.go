package solver

import (
	"math"
	"testing"

	"richards-service/internal/constitutive"
)

// fineSand / coarseSand are the two test materials. The coarse sand has a
// higher Ks but a much higher air-entry alpha and pore index, so at the
// same unsaturated pressure head it is far LESS conductive than the fine
// sand — the capillary-barrier contrast the layered tests rely on.
func fineSand() constitutive.Params {
	return constitutive.Params{Alpha: 7.0, N: 2.2, ThetaR: 0.05, ThetaS: 0.41, Ks: 6e-5}
}

func coarseSand() constitutive.Params {
	return constitutive.Params{Alpha: 14.0, N: 2.6, ThetaR: 0.04, ThetaS: 0.43, Ks: 1.2e-4}
}

// layeredTestProfile builds the 0.4 m fine-over-0.6 m coarse profile with
// 24 + 36 cells (both segment cell thicknesses are exactly 1/60 m, so the
// grid is uniform overall and directly comparable to NewGrid(60, 1)).
func layeredTestProfile() *Profile {
	return &Profile{Segments: []Segment{
		{Thickness: 0.4, NZ: 24, Params: fineSand()},
		{Thickness: 0.6, NZ: 36, Params: coarseSand()},
	}}
}

func layeredTestGrid(t *testing.T) Grid {
	t.Helper()
	g, err := NewLayeredGrid([]float64{0.4, 0.6}, []int{24, 36})
	if err != nil {
		t.Fatalf("layered grid: %v", err)
	}
	return g
}

func layeredPondingSolver(t *testing.T, theta0 float64) *Solver {
	t.Helper()
	prof := layeredTestProfile()
	g := layeredTestGrid(t)
	s, err := NewLayeredSolverFromTheta(prof, g, TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(theta0, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatalf("build layered solver: %v", err)
	}
	return s
}

// layeredFrontDepth is frontDepth with a per-segment saturation threshold.
func layeredFrontDepth(s *Solver, theta []float64, th0, frac float64) float64 {
	depth := -1.0
	for i, th := range theta {
		seg := s.Profile.Segments[s.Grid.MaterialIndex(i)]
		thr := th0 + frac*(seg.Params.ThetaS-th0)
		if th > thr {
			depth = s.Grid.Z[i]
		}
	}
	return depth
}

func TestLayeredGridInterfacesAlignWithFaces(t *testing.T) {
	g := layeredTestGrid(t)
	if g.NZ != 60 {
		t.Fatalf("NZ=%d want 60", g.NZ)
	}
	if math.Abs(g.Depth-1.0) > 1e-15 {
		t.Fatalf("depth %v", g.Depth)
	}
	// The material interface at 0.4 m must coincide exactly with face 24.
	if got := g.ZFace[24]; got != 0.4 {
		t.Fatalf("interface face depth %v, want exactly 0.4", got)
	}
	for i := 0; i < 24; i++ {
		if g.MaterialIndex(i) != 0 {
			t.Fatalf("cell %d mapped to segment %d", i, g.MaterialIndex(i))
		}
	}
	for i := 24; i < 60; i++ {
		if g.MaterialIndex(i) != 1 {
			t.Fatalf("cell %d mapped to segment %d", i, g.MaterialIndex(i))
		}
	}
	// No cell straddles the interface: cell 23 ends at 0.4, cell 24 starts there.
	if math.Abs(g.Z[23]+0.5*g.Dzs[23]-0.4) > 1e-15 ||
		math.Abs(g.Z[24]-0.5*g.Dzs[24]-0.4) > 1e-15 {
		t.Fatalf("cells straddle the interface: z23=%v dz23=%v z24=%v dz24=%v",
			g.Z[23], g.Dzs[23], g.Z[24], g.Dzs[24])
	}
}

func TestLayeredGridNonUniformThickness(t *testing.T) {
	g, err := NewLayeredGrid([]float64{0.5, 0.5}, []int{25, 20})
	if err != nil {
		t.Fatal(err)
	}
	if g.Uniform {
		t.Fatal("grid with 0.02/0.025 m cells must not be flagged uniform")
	}
	if got := g.ZFace[25]; got != 0.5 {
		t.Fatalf("interface face %v want exactly 0.5", got)
	}
	if math.Abs(g.Dzs[0]-0.02) > 1e-15 || math.Abs(g.Dzs[25]-0.025) > 1e-15 {
		t.Fatalf("cell thicknesses %v %v", g.Dzs[0], g.Dzs[25])
	}
	// Faces are continuous across the interface.
	if math.Abs(g.ZFace[25]-(g.Z[24]+0.5*g.Dzs[24])) > 1e-15 {
		t.Fatal("upper segment does not end at the interface")
	}
	if math.Abs(g.ZFace[25]-(g.Z[25]-0.5*g.Dzs[25])) > 1e-15 {
		t.Fatal("lower segment does not start at the interface")
	}
}

func TestLayeredGridRejectsBadSegments(t *testing.T) {
	if _, err := NewLayeredGrid([]float64{0.4, -0.1}, []int{4, 4}); err == nil {
		t.Fatal("negative segment thickness accepted")
	}
	if _, err := NewLayeredGrid([]float64{0.4, 0.6}, []int{4, 0}); err == nil {
		t.Fatal("zero-cell segment accepted")
	}
	if _, err := NewLayeredGrid(nil, nil); err == nil {
		t.Fatal("empty profile accepted")
	}
}

// TestLayeredDegenerateBitIdenticalToSingleMaterial is the bottom-line
// guarantee: a two-segment profile whose segments carry identical
// parameters must march bit-for-bit like the single-material solver.
func TestLayeredDegenerateBitIdenticalToSingleMaterial(t *testing.T) {
	p := fineSand()
	prof := &Profile{Segments: []Segment{
		{Thickness: 0.4, NZ: 24, Params: p},
		{Thickness: 0.6, NZ: 36, Params: p},
	}}
	g := layeredTestGrid(t)
	if !g.Uniform {
		t.Fatal("equal-thickness segments should collapse to the uniform grid")
	}
	layered, err := NewLayeredSolverFromTheta(prof, g, TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(0.12, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	ref, err := NewSolverFromTheta(p, NewGrid(60, 1.0), TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(0.12, 60), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	for k := 0; k < 60; k++ {
		rl, err := layered.StepAdaptive(60, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("layered interval %d: %v", k, err)
		}
		rr, err := ref.StepAdaptive(60, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("reference interval %d: %v", k, err)
		}
		for i := range rl.ThetaAfter {
			if rl.ThetaAfter[i] != rr.ThetaAfter[i] || rl.HAfter[i] != rr.HAfter[i] {
				t.Fatalf("interval %d layer %d: layered (%g,%g) != uniform (%g,%g)",
					k, i, rl.ThetaAfter[i], rl.HAfter[i], rr.ThetaAfter[i], rr.HAfter[i])
			}
		}
		if rl.StorageAfter != rr.StorageAfter || rl.CumTopFlux != rr.CumTopFlux {
			t.Fatalf("interval %d accounting differs", k)
		}
	}
}

// TestLayeredDegenerateNonUniformGrid checks the general (non-uniform)
// discretisation path: identical parameters in both segments must behave
// exactly like the single-material solver on the same non-uniform grid.
func TestLayeredDegenerateNonUniformGrid(t *testing.T) {
	p := fineSand()
	prof := &Profile{Segments: []Segment{
		{Thickness: 0.5, NZ: 25, Params: p},
		{Thickness: 0.5, NZ: 20, Params: p},
	}}
	g, err := NewLayeredGrid([]float64{0.5, 0.5}, []int{25, 20})
	if err != nil {
		t.Fatal(err)
	}
	if g.Uniform {
		t.Fatal("expected non-uniform grid")
	}
	layered, err := NewLayeredSolverFromTheta(prof, g, TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(0.12, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	// Single-material solver on the very same grid: identical code path.
	ref, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(0.12, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	for k := 0; k < 30; k++ {
		rl, err := layered.Step(60)
		if err != nil {
			t.Fatalf("layered step %d: %v", k, err)
		}
		rr, err := ref.Step(60)
		if err != nil {
			t.Fatalf("reference step %d: %v", k, err)
		}
		for i := range rl.ThetaAfter {
			if rl.ThetaAfter[i] != rr.ThetaAfter[i] {
				t.Fatalf("step %d layer %d: %g != %g", k, i,
					rl.ThetaAfter[i], rr.ThetaAfter[i])
			}
		}
		if math.Abs(rl.MassBalanceResidual) > 1e-10 {
			t.Fatalf("non-uniform grid closure %g", rl.MassBalanceResidual)
		}
	}
}

// TestLayeredMassClosureAcrossInterfaces: the whole-column balance must
// close to the same precision as the homogeneous solver even though the
// front crosses a material interface.
func TestLayeredMassClosureAcrossInterfaces(t *testing.T) {
	s := layeredPondingSolver(t, 0.12)
	steps, err := s.MarchAdaptive(60, 120, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	// Confirm the front actually crossed the interface during the run.
	if got := layeredFrontDepth(s, steps[len(steps)-1].ThetaAfter, 0.12, 0.3); got < 0.45 {
		t.Fatalf("front never crossed the interface (depth %v)", got)
	}
	for k := range steps {
		if math.Abs(steps[k].MassBalanceResidual) > 1e-9 {
			t.Fatalf("interval %d closure residual %g", k, steps[k].MassBalanceResidual)
		}
	}
	tot := s.Storage() - steps[0].StorageBefore - (s.CumTop - s.CumBot)
	if math.Abs(tot) > 1e-9 {
		t.Fatalf("global closure %g", tot)
	}
}

// TestLayeredSegmentMassBalanceAndInterfaceFlux: each material segment
// individually must satisfy its own balance dS_seg = (q_in - q_out)*dt
// with the shared interface flux; the flux entering the interface from
// the fine side equals the flux leaving it into the coarse side.
func TestLayeredSegmentMassBalanceAndInterfaceFlux(t *testing.T) {
	s := layeredPondingSolver(t, 0.12)
	iface := 24 // face index of the material interface
	steps, err := s.MarchAdaptive(60, 90, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	for k := range steps {
		st := &steps[k]
		if len(st.FaceFluxes) != s.Grid.NZ+1 {
			t.Fatalf("interval %d: face fluxes not reported", k)
		}
		qIface := st.FaceFluxes[iface]
		// Segment 0: cells 0..23, inflow q_top, outflow q_iface.
		dS0, dS1 := 0.0, 0.0
		for i := 0; i < iface; i++ {
			dS0 += (st.ThetaAfter[i] - st.ThetaBefore[i]) * s.Grid.Dzs[i]
		}
		for i := iface; i < s.Grid.NZ; i++ {
			dS1 += (st.ThetaAfter[i] - st.ThetaBefore[i]) * s.Grid.Dzs[i]
		}
		res0 := dS0 - (st.FaceFluxes[0]-qIface)*st.Dt
		res1 := dS1 - (qIface-st.FaceFluxes[s.Grid.NZ])*st.Dt
		if math.Abs(res0) > 1e-10 {
			t.Fatalf("interval %d segment 0 balance residual %g", k, res0)
		}
		if math.Abs(res1) > 1e-10 {
			t.Fatalf("interval %d segment 1 balance residual %g", k, res1)
		}
	}
}

// TestLayeredInterfaceStateUsesOwnCurves verifies that the two cells
// flanking the material interface each follow their own retention and
// conductivity curves, that theta jumps across the interface, and that
// the interface flux reconstructed independently from the two sides' own
// curves matches the solver's face flux exactly.
func TestLayeredInterfaceStateUsesOwnCurves(t *testing.T) {
	s := layeredPondingSolver(t, 0.12)
	steps, err := s.MarchAdaptive(60, 90, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	p0, p1 := fineSand(), coarseSand()
	up, down := 23, 24
	lUp := 0.5 * s.Grid.Dzs[up]
	lDown := 0.5 * s.Grid.Dzs[down]
	lFace := lUp + lDown
	checked := 0
	for k := range steps {
		st := &steps[k]
		hU, hD := st.HAfter[up], st.HAfter[down]
		thU, thD := st.ThetaAfter[up], st.ThetaAfter[down]
		// Each side's water content must be exactly its own curve's value.
		if thU != p0.WaterContent(hU) {
			t.Fatalf("interval %d: upper theta not on fine-sand curve", k)
		}
		if thD != p1.WaterContent(hD) {
			t.Fatalf("interval %d: lower theta not on coarse-sand curve", k)
		}
		// Each side inside its own legal range.
		if thU < p0.ThetaR-1e-12 || thU > p0.ThetaS+1e-12 ||
			thD < p1.ThetaR-1e-12 || thD > p1.ThetaS+1e-12 {
			t.Fatalf("interval %d: interface theta out of segment range", k)
		}
		// Reconstruct the face flux from the two sides' own K(h) curves.
		kU := p0.Conductivity(hU)
		kD := p1.Conductivity(hD)
		kf := weightedFaceK(kU, kD, lUp, lDown)
		q := kf + (kf/lFace)*(hU-hD)
		if q != st.FaceFluxes[24] {
			t.Fatalf("interval %d: reconstructed interface flux %g != reported %g",
				k, q, st.FaceFluxes[24])
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no intervals checked")
	}
	// Across the interface the water content jumps (different curves).
	last := steps[len(steps)-1]
	if last.ThetaAfter[up] == last.ThetaAfter[down] {
		t.Fatal("theta should differ across the material interface")
	}
}

// TestLayeredFrontHesitatesAtInterface: the wetting front must observably
// change speed when crossing from the fine into the coarse layer, and lag
// behind the homogeneous fine-sand reference once it reaches the
// interface (capillary barrier).
func TestLayeredFrontHesitatesAtInterface(t *testing.T) {
	const th0 = 0.12
	s := layeredPondingSolver(t, th0)
	// Homogeneous fine-sand reference on the identical uniform grid.
	ref, err := NewSolverFromTheta(fineSand(), NewGrid(60, 1.0), TopPondedHead,
		0.02, BottomFreeDrainage, repeat(th0, 60), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}

	nInt := 120
	frontL := make([]float64, nInt)
	frontR := make([]float64, nInt)
	times := make([]float64, nInt)
	for k := 0; k < nInt; k++ {
		aggL, err := s.StepAdaptive(60, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("layered interval %d: %v", k, err)
		}
		aggR, err := ref.StepAdaptive(60, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("reference interval %d: %v", k, err)
		}
		frontL[k] = layeredFrontDepth(s, aggL.ThetaAfter, th0, 0.3)
		frontR[k] = frontDepth(aggR.ThetaAfter, th0, fineSand().ThetaS, 0.3, ref.Grid)
		times[k] = aggL.TimeAfter
	}

	// Time for the layered front to traverse 0.25->0.35 (inside fine) vs
	// 0.35->0.45 (crossing the interface at 0.4).
	timeToReach := func(fronts []float64, z float64) float64 {
		for k, f := range fronts {
			if f >= z {
				return times[k]
			}
		}
		return math.Inf(1)
	}
	tFine := timeToReach(frontL, 0.35) - timeToReach(frontL, 0.25)
	tCross := timeToReach(frontL, 0.45) - timeToReach(frontL, 0.35)
	if !(tCross > 1.5*tFine) {
		t.Fatalf("no observable slowdown at interface: t(0.25-0.35)=%gs t(0.35-0.45)=%gs",
			tFine, tCross)
	}
	// The layered front must lag the homogeneous reference once the
	// reference is well past the interface.
	kRef := -1
	for k, f := range frontR {
		if f >= 0.55 {
			kRef = k
			break
		}
	}
	if kRef < 0 {
		t.Fatal("reference front never reached 0.55 m")
	}
	if !(frontL[kRef] < frontR[kRef]-0.05) {
		t.Fatalf("layered front %v not lagging reference %v at t=%v",
			frontL[kRef], frontR[kRef], times[kRef])
	}
}

// TestLayeredHydrostaticPerMaterialCurves: a hydrostatic initial profile
// in a layered column must evaluate each cell's water content with its
// own segment's retention curve, and stay motionless under zero flux.
func TestLayeredHydrostaticPerMaterialCurves(t *testing.T) {
	prof := layeredTestProfile()
	g := layeredTestGrid(t)
	const zwt = 1.5
	heads := make([]float64, g.NZ)
	for i := range heads {
		heads[i] = g.Z[i] - zwt
	}
	s, err := NewLayeredSolver(prof, g, TopZeroFlux, 0, BottomZeroFlux, heads,
		DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	p0, p1 := fineSand(), coarseSand()
	for i := 0; i < g.NZ; i++ {
		want := p0.WaterContent(heads[i])
		if g.MaterialIndex(i) == 1 {
			want = p1.WaterContent(heads[i])
		}
		if s.Theta[i] != want {
			t.Fatalf("cell %d theta %g, want %g from its own segment curve",
				i, s.Theta[i], want)
		}
	}
	// The two materials give different water contents at the same head:
	// cells just above/below the interface have nearly equal heads but
	// must differ in theta.
	if math.Abs(s.Theta[23]-s.Theta[24]) < 1e-3 {
		t.Fatalf("hydrostatic theta should jump at interface: %g vs %g",
			s.Theta[23], s.Theta[24])
	}
	before := append([]float64(nil), s.Theta...)
	if _, err := s.MarchAdaptive(300, 24, DefaultAdaptiveConfig()); err != nil {
		t.Fatalf("march: %v", err)
	}
	for i := range before {
		if math.Abs(s.Theta[i]-before[i]) > 1e-12 {
			t.Fatalf("layer %d drifted: %.12f -> %.12f", i, before[i], s.Theta[i])
		}
	}
	if math.Abs(s.CumTop)+math.Abs(s.CumBot) > 0 {
		t.Fatal("zero-flux boundaries produced flux")
	}
}

// TestLayeredThetaValidationPerSegment: the same numerical water content
// is legal in the coarse segment but out of range in the fine one; the
// bounds check must follow the owning segment.
func TestLayeredThetaValidationPerSegment(t *testing.T) {
	prof := layeredTestProfile()
	g := layeredTestGrid(t)
	// 0.045 is above coarse thetaR (0.04) but below fine thetaR (0.05).
	th := repeat(0.045, g.NZ)
	_, err := NewLayeredSolverFromTheta(prof, g, TopZeroFlux, 0, BottomZeroFlux,
		th, DefaultOptions())
	if err == nil {
		t.Fatal("theta=0.045 in fine segment (thetaR=0.05) must be rejected")
	}
	// Legal in the coarse cells only.
	for i := 24; i < g.NZ; i++ {
		th[i] = 0.045
	}
	for i := 0; i < 24; i++ {
		th[i] = 0.12
	}
	if _, err := NewLayeredSolverFromTheta(prof, g, TopZeroFlux, 0, BottomZeroFlux,
		th, DefaultOptions()); err != nil {
		t.Fatalf("per-segment legal profile rejected: %v", err)
	}
	// 0.42 exceeds fine thetaS (0.41) but is legal for coarse (0.43).
	th2 := repeat(0.12, g.NZ)
	th2[10] = 0.42
	if _, err := NewLayeredSolverFromTheta(prof, g, TopZeroFlux, 0, BottomZeroFlux,
		th2, DefaultOptions()); err == nil {
		t.Fatal("theta=0.42 in fine segment (thetaS=0.41) must be rejected")
	}
	th2[10] = 0.12
	th2[40] = 0.42
	if _, err := NewLayeredSolverFromTheta(prof, g, TopZeroFlux, 0, BottomZeroFlux,
		th2, DefaultOptions()); err != nil {
		t.Fatalf("theta=0.42 in coarse segment (thetaS=0.43) must be accepted: %v", err)
	}
}

// TestLayeredStrongContrastConverges: several orders of magnitude between
// the two segments' saturated conductivities must not stall the Newton
// iteration, and mass closure must hold at the usual precision.
func TestLayeredStrongContrastConverges(t *testing.T) {
	prof := &Profile{Segments: []Segment{
		{Thickness: 0.4, NZ: 24, Params: constitutive.Params{
			Alpha: 5.0, N: 2.0, ThetaR: 0.06, ThetaS: 0.40, Ks: 2e-6}},
		{Thickness: 0.6, NZ: 36, Params: constitutive.Params{
			Alpha: 18.0, N: 2.8, ThetaR: 0.03, ThetaS: 0.44, Ks: 2e-3}},
	}}
	g, err := NewLayeredGrid([]float64{0.4, 0.6}, []int{24, 36})
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewLayeredSolverFromTheta(prof, g, TopPondedHead, 0.02,
		BottomFreeDrainage, repeat(0.10, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	steps, err := s.MarchAdaptive(60, 60, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("strong-contrast march failed to converge: %v", err)
	}
	for k := range steps {
		if math.Abs(steps[k].MassBalanceResidual) > 1e-9 {
			t.Fatalf("interval %d closure %g", k, steps[k].MassBalanceResidual)
		}
	}
}

// TestWeightedFaceKReducesToHarmonic: with equal half-distances the
// weighted interface conductivity equals the plain harmonic mean.
func TestWeightedFaceKReducesToHarmonic(t *testing.T) {
	for _, k := range [][2]float64{{1e-8, 5e-5}, {3e-6, 3e-6}, {1e-3, 1e-9}} {
		w := weightedFaceK(k[0], k[1], 0.5, 0.5)
		h := HarmonicMean.mean(k[0], k[1])
		if w != h {
			t.Fatalf("weighted %g != harmonic %g for K=%v", w, h, k)
		}
	}
	// Unequal distances: series resistance of the two half cells.
	// K = L / (l1/K1 + l2/K2).
	got := weightedFaceK(1e-4, 1e-6, 0.75, 0.25)
	want := 1.0 / (0.75/1e-4 + 0.25/1e-6)
	if math.Abs(got-want) > 1e-18*want {
		t.Fatalf("weighted face K %g want %g", got, want)
	}
}

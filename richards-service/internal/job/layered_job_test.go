package job

import (
	"errors"
	"math"
	"testing"

	"richards-service/internal/solver"
)

func fineMat() Material {
	return Material{Alpha: 4.5, N: 2.68, ThetaR: 0.045, ThetaS: 0.43, Ks: 1.7e-5}
}
func coarseMat() Material {
	return Material{Alpha: 10.0, N: 2.68, ThetaR: 0.02, ThetaS: 0.36, Ks: 1.2e-4}
}

func layer(thickness float64, nz int, m Material) MaterialLayer {
	return MaterialLayer{Thickness: thickness, NumLayers: nz,
		Alpha: m.Alpha, N: m.N, ThetaR: m.ThetaR, ThetaS: m.ThetaS, Ks: m.Ks}
}

// twoLayerRequest builds a valid fine-over-coarse layered ponding request,
// initialised by uniform water content (each value chosen inside its own
// segment's bounds).
func twoLayerRequest() Request {
	th := make([]float64, 50)
	for i := 0; i < 20; i++ {
		th[i] = 0.12 // fine [0.045, 0.43]
	}
	for i := 20; i < 50; i++ {
		th[i] = 0.05 // coarse [0.02, 0.36]
	}
	return Request{
		Column: Column{Thickness: 1.0, NZ: 50},
		Materials: []MaterialLayer{
			layer(0.4, 20, fineMat()),
			layer(0.6, 30, coarseMat()),
		},
		Initial:  Initial{Kind: "water_content", WaterContent: th},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 10800, StepSize: 120},
	}
}

// identical-layer request reproduces the homogeneous request: two segments
// with exactly equal parameters (interface at 0.5 m on a face).
func twoIdenticalLayerRequest(mat Material) Request {
	th := make([]float64, 40)
	for i := range th {
		th[i] = 0.15
	}
	return Request{
		Column: Column{Thickness: 1.0, NZ: 40},
		Materials: []MaterialLayer{
			layer(0.5, 20, mat),
			layer(0.5, 20, mat),
		},
		Initial:  Initial{Kind: "water_content", WaterContent: th},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 5400, StepSize: 30},
	}
}

func TestLayeredRunMassClosureAndEcho(t *testing.T) {
	r := twoLayerRequest()
	res, err := Run("L", r)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Materials) != 2 {
		t.Fatalf("echoed materials=%d want 2", len(res.Materials))
	}
	if res.Materials[0].Thickness != 0.4 || res.Materials[1].Thickness != 0.6 {
		t.Fatalf("segment thicknesses: %v %v", res.Materials[0].Thickness, res.Materials[1].Thickness)
	}
	if res.Materials[0].CellStart != 0 || res.Materials[0].CellEnd != 20 ||
		res.Materials[1].CellStart != 20 || res.Materials[1].CellEnd != 50 {
		t.Fatalf("segment cell ranges: %+v %+v", res.Materials[0], res.Materials[1])
	}
	if got := res.Grid.SegmentInterfaceDepthsM; len(got) != 3 ||
		got[0] != 0 || got[1] != 0.4 || got[2] != 1.0 {
		t.Fatalf("interface depths %v", got)
	}
	if res.Grid.SegmentOfCell[19] != 0 || res.Grid.SegmentOfCell[20] != 1 {
		t.Fatalf("segment assignment wrong at interface")
	}
	for k, st := range res.Steps {
		if math.Abs(st.MassBalanceResidM) > 1e-9 {
			t.Fatalf("interval %d closure %g", k, st.MassBalanceResidM)
		}
		// every cell stays inside its own segment's bounds
		for _, l := range st.Layers {
			var p Material
			if l.Segment == 0 {
				p = fineMat()
			} else {
				p = coarseMat()
			}
			if l.WaterContent < p.ThetaR-1e-12 || l.WaterContent > p.ThetaS+1e-12 {
				t.Fatalf("interval %d cell theta %g out of segment %d bounds",
					k, l.WaterContent, l.Segment)
			}
		}
	}
	if math.Abs(res.TotalMassResidualM) > 1e-9 {
		t.Fatalf("total layered closure %g", res.TotalMassResidualM)
	}
}

// Degradation at the job level: splitting a homogeneous column into two
// segments carrying identical parameters gives the same profiles at every
// output time as the legacy single-material request.
func TestIdenticalLayersMatchHomogeneousJob(t *testing.T) {
	hom := baseRequest()
	layered := twoIdenticalLayerRequest(*hom.Material)
	rh, err := Run("hom", hom)
	if err != nil {
		t.Fatal(err)
	}
	rl, err := Run("lay", layered)
	if err != nil {
		t.Fatal(err)
	}
	if len(rh.Steps) != len(rl.Steps) {
		t.Fatalf("step count %d vs %d", len(rh.Steps), len(rl.Steps))
	}
	for k := range rh.Steps {
		for i := range rh.Steps[k].Layers {
			a := rh.Steps[k].Layers[i]
			b := rl.Steps[k].Layers[i]
			if a.WaterContent != b.WaterContent || a.PressureHead != b.PressureHead {
				t.Fatalf("step %d layer %d differs: hom th=%g h=%g ; layered th=%g h=%g",
					k, i, a.WaterContent, a.PressureHead, b.WaterContent, b.PressureHead)
			}
		}
		if rh.Steps[k].MassBalanceResidM != rl.Steps[k].MassBalanceResidM {
			t.Fatalf("step %d closure differs: %v vs %v",
				k, rh.Steps[k].MassBalanceResidM, rl.Steps[k].MassBalanceResidM)
		}
	}
	if math.Abs(rh.FinalStorageM-rl.FinalStorageM) > 0 {
		t.Fatalf("final storage differs: %v vs %v", rh.FinalStorageM, rl.FinalStorageM)
	}
}

// single-step and full-run over a layered column from the same initial state
// give identical first-interval profiles. The very first interval from the
// smooth initial state converges without internal substepping (Substeps==1),
// so the strict single-step solve and the adaptive march are one and the
// same Newton solve.
func TestLayeredSingleStepMatchesFullRun(t *testing.T) {
	r := twoLayerRequest()
	r.Time = TimeSpec{TotalTime: 30, StepSize: 30}
	full, err := Run("full", r)
	if err != nil {
		t.Fatal(err)
	}
	if full.Steps[0].Substeps != 1 {
		t.Fatalf("setup: expected single substep, got %d", full.Steps[0].Substeps)
	}
	stepReq := StepRequest{
		Column:    r.Column,
		Materials: r.Materials,
		Initial:   r.Initial,
		Boundary:  r.Boundary,
		StepSize:  30,
	}
	one, err := RunStep("one", stepReq)
	if err != nil {
		t.Fatal(err)
	}
	if len(one.Materials) != 2 {
		t.Fatalf("single-step materials echo=%d", len(one.Materials))
	}
	for i := range full.Steps[0].Layers {
		a, b := full.Steps[0].Layers[i], one.After[i]
		if math.Abs(a.WaterContent-b.WaterContent) > 1e-13 ||
			math.Abs(a.PressureHead-b.PressureHead) > 1e-12 {
			t.Fatalf("layer %d differs: full th=%g h=%g ; step th=%g h=%g",
				i, a.WaterContent, a.PressureHead, b.WaterContent, b.PressureHead)
		}
		if a.Segment != b.Segment {
			t.Fatalf("layer %d segment tag differs", i)
		}
	}
	if math.Abs(full.Steps[0].MassBalanceResidM-one.Step.MassBalanceResidM) > 1e-14 {
		t.Fatal("single-step layered closure differs from full-run first interval")
	}
}

func TestLayeredValidationRejectsBadProfiles(t *testing.T) {
	cases := []struct {
		name string
		mut  func(r *Request)
		code string
	}{
		{"segment n<=1", func(r *Request) { r.Materials[1].N = 1 }, CodeNInvalid},
		{"segment alpha<=0", func(r *Request) { r.Materials[0].Alpha = 0 }, CodeAlphaNonPositive},
		{"segment thetaR>=thetaS", func(r *Request) { r.Materials[1].ThetaR = 0.5 }, CodeThetaRangeInvalid},
		{"segment ks<=0", func(r *Request) { r.Materials[0].Ks = 0 }, CodeKsNonPositive},
		{"segment thickness<=0", func(r *Request) { r.Materials[0].Thickness = 0 }, CodeMaterialInvalid},
		{"thickness sum mismatch", func(r *Request) { r.Materials[1].Thickness = 0.5 }, CodeMaterialInvalid},
		{"no material at all", func(r *Request) { r.Material = nil; r.Materials = nil }, CodeMaterialInvalid},
		{"both material and materials", func(r *Request) { m := fineMat(); r.Material = &m }, CodeMaterialInvalid},
		{"cell counts do not total", func(r *Request) { r.Materials[0].NumLayers = 19 }, CodeMaterialInvalid},
		{"more segments than cells", func(r *Request) {
			r.Column.NZ = 1
			r.Materials[0].NumLayers = 0
			r.Materials[1].NumLayers = 0
		}, CodeMaterialInvalid},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := twoLayerRequest()
			tc.mut(&r)
			_, err := Run("j", r)
			if err == nil {
				t.Fatal("expected validation error, got nil")
			}
			if got := codeOf(t, err); got != tc.code {
				t.Fatalf("code=%s want %s (%v)", got, tc.code, err)
			}
		})
	}
}

// the error message identifies the offending segment.
func TestLayeredValidationNamesSegment(t *testing.T) {
	r := twoLayerRequest()
	r.Materials[1].Ks = -3
	_, err := Run("j", r)
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("want ValidationError, got %v", err)
	}
	if ve.Field != "materials[1].ks" {
		t.Fatalf("field=%q want materials[1].ks", ve.Field)
	}
}

// per-segment initial water-content bounds: a value legal in the fine
// segment but above the coarse segment's thetaS is rejected when placed in a
// coarse cell, and accepted in a fine cell.
func TestLayeredInitialThetaBoundsPerSegment(t *testing.T) {
	r := twoLayerRequest()
	r.Initial.WaterContent[30] = 0.40 // > coarse thetaS 0.36
	if _, err := Run("j", r); err == nil {
		t.Fatal("expected per-segment bounds rejection in coarse cell")
	} else if codeOf(t, err) != CodeProfileInvalid {
		t.Fatalf("code %v", err)
	}
	// same value in a fine cell is legal.
	r = twoLayerRequest()
	r.Initial.WaterContent[30] = 0.05
	r.Initial.WaterContent[5] = 0.40
	if _, err := Run("j", r); err != nil {
		t.Fatalf("0.40 should be legal in fine segment: %v", err)
	}
}

// hydrostatic initialisation over a layered column: heads are geometric but
// thetas come from each cell's own retention curve; the zero-flux column
// stays in equilibrium.
func TestLayeredHydrostaticEquilibrium(t *testing.T) {
	r := Request{
		Column: Column{Thickness: 1.0, NZ: 50},
		Materials: []MaterialLayer{
			layer(0.4, 20, fineMat()),
			layer(0.6, 30, coarseMat()),
		},
		Initial:  Initial{Kind: "hydrostatic", WaterTableDepthM: 1.5},
		Boundary: Boundary{Top: solver.TopZeroFlux, Bottom: solver.BottomZeroFlux},
		Time:     TimeSpec{TotalTime: 7200, StepSize: 300},
	}
	res, err := Run("h", r)
	if err != nil {
		t.Fatal(err)
	}
	first, last := res.Steps[0], res.Steps[len(res.Steps)-1]
	for i := range first.Layers {
		if math.Abs(first.Layers[i].WaterContent-last.Layers[i].WaterContent) > 1e-10 {
			t.Fatalf("layer %d drifted", i)
		}
	}
	if math.Abs(res.TotalMassResidualM) > 1e-10 {
		t.Fatalf("layered hydrostatic closure %g", res.TotalMassResidualM)
	}
	// Theta at equal head differs across the interface (different curves).
	if first.Layers[19].WaterContent == first.Layers[20].WaterContent {
		t.Fatal("hydrostatic theta unexpectedly equal across the interface")
	}
}

// omitting per-segment cell counts allocates cells proportionally and the
// total still matches column.num_layers.
func TestLayeredImplicitCellAllocation(t *testing.T) {
	r := twoLayerRequest()
	r.Materials[0].NumLayers = 0
	r.Materials[1].NumLayers = 0
	res, err := Run("j", r)
	if err != nil {
		t.Fatal(err)
	}
	tot := 0
	for _, m := range res.Materials {
		tot += m.CellEnd - m.CellStart
	}
	if tot != 50 {
		t.Fatalf("allocated %d cells want 50", tot)
	}
	// the 0.4/0.6 split should allocate ~20/30
	if res.Materials[0].CellEnd-res.Materials[0].CellStart != 20 ||
		res.Materials[1].CellEnd-res.Materials[1].CellStart != 30 {
		t.Fatalf("allocation %d/%d", res.Materials[0].CellEnd, res.Materials[1].CellEnd-res.Materials[1].CellStart)
	}
	if got := res.Grid.SegmentInterfaceDepthsM[1]; got != 0.4 {
		t.Fatalf("interface depth %v", got)
	}
}

// a single-segment layered form also reproduces the homogeneous run (the
// solver detects the bitwise-uniform geometry and takes the identical path).
func TestSingleSegmentLayeredMatchesHomogeneousJob(t *testing.T) {
	hom := baseRequest()
	th := make([]float64, 40)
	for i := range th {
		th[i] = 0.15
	}
	oneSeg := Request{
		Column:    Column{Thickness: 1.0, NZ: 40},
		Materials: []MaterialLayer{layer(1.0, 40, *hom.Material)},
		Initial:   Initial{Kind: "water_content", WaterContent: th},
		Boundary:  hom.Boundary,
		Time:      hom.Time,
	}
	rh, err := Run("hom", hom)
	if err != nil {
		t.Fatal(err)
	}
	rs, err := Run("seg", oneSeg)
	if err != nil {
		t.Fatal(err)
	}
	for k := range rh.Steps {
		for i := range rh.Steps[k].Layers {
			a := rh.Steps[k].Layers[i]
			b := rs.Steps[k].Layers[i]
			if a.WaterContent != b.WaterContent || a.PressureHead != b.PressureHead {
				t.Fatalf("step %d layer %d differs", k, i)
			}
		}
	}
	if len(rs.Materials) != 1 || rs.Materials[0].CellEnd != 40 {
		t.Fatalf("single-segment echo wrong: %+v", rs.Materials)
	}
}

// legacy homogeneous input form still works without any materials wrapper.
func TestLegacyHomogeneousFormStillAccepted(t *testing.T) {
	r := baseRequest()
	if r.Material == nil {
		t.Fatal("baseline uses legacy form")
	}
	res, err := Run("j", r)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Materials) != 1 {
		t.Fatalf("homogeneous job should echo one material, got %d", len(res.Materials))
	}
	if res.Materials[0].Alpha != r.Material.Alpha {
		t.Fatal("echoed material mismatch")
	}
}

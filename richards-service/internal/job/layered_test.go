package job

import (
	"errors"
	"math"
	"strings"
	"testing"

	"richards-service/internal/solver"
)

// layeredBaseRequest is the fine-over-coarse layered job used across the
// job-level tests.
func layeredBaseRequest() Request {
	return Request{
		Column: Column{Thickness: 1.0, NZ: 60},
		Profile: &SoilProfileRequest{Layers: []SoilLayer{
			{Thickness: 0.4, NumLayers: 24,
				Material: Material{Alpha: 7.0, N: 2.2, ThetaR: 0.05, ThetaS: 0.41, Ks: 6e-5}},
			{Thickness: 0.6, NumLayers: 36,
				Material: Material{Alpha: 14.0, N: 2.6, ThetaR: 0.04, ThetaS: 0.43, Ks: 1.2e-4}},
		}},
		Initial:  Initial{Kind: "water_content", WaterContent: repeatSlice(0.12, 60)},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 3600, StepSize: 60},
	}
}

func TestLayeredRunAcceptedEchoesProfileAndCloses(t *testing.T) {
	res, err := Run("lay", layeredBaseRequest())
	if err != nil {
		t.Fatal(err)
	}
	if !res.MaterialConfig.Layered {
		t.Fatal("material_config must report a layered profile")
	}
	if len(res.MaterialConfig.Segments) != 2 {
		t.Fatalf("segments=%d want 2", len(res.MaterialConfig.Segments))
	}
	seg0 := res.MaterialConfig.Segments[0]
	if seg0.ThicknessM != 0.4 || seg0.NumLayers != 24 || seg0.Alpha != 7.0 ||
		seg0.Ks != 6e-5 {
		t.Fatalf("segment 0 echo wrong: %+v", seg0)
	}
	if math.Abs(seg0.MLocked-(1-1/2.2)) > 1e-15 {
		t.Fatalf("segment 0 m_locked %g", seg0.MLocked)
	}
	if !res.Grid.Layered {
		t.Fatal("grid must be flagged layered")
	}
	if len(res.Grid.InterfaceFaces) != 1 || res.Grid.InterfaceFaces[0] != 24 {
		t.Fatalf("interface faces %v, want [24]", res.Grid.InterfaceFaces)
	}
	if got := res.Grid.FaceDepthsM[24]; got != 0.4 {
		t.Fatalf("interface face depth %v want 0.4", got)
	}
	if len(res.Grid.SegmentOf) != 60 || res.Grid.SegmentOf[0] != 0 || res.Grid.SegmentOf[59] != 1 {
		t.Fatalf("segment ownership echo wrong")
	}
	for k, st := range res.Steps {
		if math.Abs(st.MassBalanceResidM) > 1e-9 {
			t.Fatalf("interval %d closure %g", k, st.MassBalanceResidM)
		}
	}
	if math.Abs(res.TotalMassResidualM) > 1e-9 {
		t.Fatalf("total closure %g", res.TotalMassResidualM)
	}
}

// TestLayeredDegenerateJobMatchesUniform: two segments with identical
// parameters must reproduce the single-material job layer by layer.
func TestLayeredDegenerateJobMatchesUniform(t *testing.T) {
	mat := Material{Alpha: 7.0, N: 2.2, ThetaR: 0.05, ThetaS: 0.41, Ks: 6e-5}
	uniform := Request{
		Column:   Column{Thickness: 1.0, NZ: 60},
		Material: mat,
		Initial:  Initial{Kind: "water_content", WaterContent: repeatSlice(0.12, 60)},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 3600, StepSize: 60},
	}
	layered := uniform
	layered.Material = Material{}
	layered.Profile = &SoilProfileRequest{Layers: []SoilLayer{
		{Thickness: 0.4, NumLayers: 24, Material: mat},
		{Thickness: 0.6, NumLayers: 36, Material: mat},
	}}
	ru, err := Run("u", uniform)
	if err != nil {
		t.Fatal(err)
	}
	rl, err := Run("l", layered)
	if err != nil {
		t.Fatal(err)
	}
	if len(ru.Steps) != len(rl.Steps) {
		t.Fatal("step count differs")
	}
	for k := range ru.Steps {
		su, sl := ru.Steps[k], rl.Steps[k]
		for i := range su.Layers {
			du := math.Abs(su.Layers[i].WaterContent - sl.Layers[i].WaterContent)
			dh := math.Abs(su.Layers[i].PressureHead - sl.Layers[i].PressureHead)
			if du > 1e-13 || dh > 1e-12 {
				t.Fatalf("interval %d layer %d: theta diff %g head diff %g", k, i, du, dh)
			}
		}
		if math.Abs(su.StorageM-sl.StorageM) > 1e-13 {
			t.Fatalf("interval %d storage differs", k)
		}
	}
	if math.Abs(ru.FinalStorageM-rl.FinalStorageM) > 1e-13 {
		t.Fatal("final storage differs")
	}
}

// TestLayeredSingleSegmentMatchesUniform: a one-segment profile is the
// degenerate case of the layered form and must behave identically.
func TestLayeredSingleSegmentMatchesUniform(t *testing.T) {
	mat := Material{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	uniform := Request{
		Column:   Column{Thickness: 1.0, NZ: 40},
		Material: mat,
		Initial:  Initial{Kind: "water_content", WaterContent: repeatSlice(0.15, 40)},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 1800, StepSize: 30},
	}
	oneSeg := uniform
	oneSeg.Material = Material{}
	oneSeg.Profile = &SoilProfileRequest{Layers: []SoilLayer{
		{Thickness: 1.0, NumLayers: 40, Material: mat},
	}}
	ru, err := Run("u", uniform)
	if err != nil {
		t.Fatal(err)
	}
	ro, err := Run("o", oneSeg)
	if err != nil {
		t.Fatal(err)
	}
	lastU := ru.Steps[len(ru.Steps)-1]
	lastO := ro.Steps[len(ro.Steps)-1]
	for i := range lastU.Layers {
		if math.Abs(lastU.Layers[i].WaterContent-lastO.Layers[i].WaterContent) > 1e-13 {
			t.Fatalf("layer %d differs between uniform and one-segment profile", i)
		}
	}
}

func TestLayeredThicknessSumMismatchRejected(t *testing.T) {
	r := layeredBaseRequest()
	r.Profile.Layers[1].Thickness = 0.5 // sums to 0.9, column is 1.0
	_, err := Run("j", r)
	if got := codeOf(t, err); got != CodeProfileSum {
		t.Fatalf("code=%s want %s", got, CodeProfileSum)
	}
	var ve *ValidationError
	_ = errors.As(err, &ve)
	if !strings.Contains(ve.Message, "0.9") {
		t.Fatalf("message should report the offending sum: %v", ve.Message)
	}
}

func TestLayeredSegmentParamsRejectedIndividually(t *testing.T) {
	cases := []struct {
		name string
		mut  func(m *Material)
	}{
		{"n <= 1", func(m *Material) { m.N = 1.0 }},
		{"alpha <= 0", func(m *Material) { m.Alpha = 0 }},
		{"thetaR >= thetaS", func(m *Material) { m.ThetaR = m.ThetaS }},
		{"ks <= 0", func(m *Material) { m.Ks = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := layeredBaseRequest()
			tc.mut(&r.Profile.Layers[1].Material)
			_, err := Run("j", r)
			if got := codeOf(t, err); got != CodeLayerParams {
				t.Fatalf("code=%s want %s", got, CodeLayerParams)
			}
			var ve *ValidationError
			_ = errors.As(err, &ve)
			if !strings.Contains(ve.Message, "segment 1") {
				t.Fatalf("error must name the offending segment: %v", ve.Message)
			}
		})
	}
}

func TestLayeredStructureViolations(t *testing.T) {
	// Zero-thickness segment.
	r := layeredBaseRequest()
	r.Profile.Layers[0].Thickness = 0
	_, err := Run("j", r)
	if got := codeOf(t, err); got != CodeLayerThickness {
		t.Fatalf("zero thickness code=%s", got)
	}
	// Empty profile.
	r = layeredBaseRequest()
	r.Profile.Layers = nil
	_, err = Run("j", r)
	if got := codeOf(t, err); got != CodeLayerCount {
		t.Fatalf("empty profile code=%s", got)
	}
	// Per-layer cell counts not summing to the column total.
	r = layeredBaseRequest()
	r.Profile.Layers[1].NumLayers = 30
	_, err = Run("j", r)
	if got := codeOf(t, err); got != CodeGridInvalid {
		t.Fatalf("cell sum mismatch code=%s", got)
	}
	// Mixed: only one segment carries a cell count.
	r = layeredBaseRequest()
	r.Profile.Layers[0].NumLayers = 0
	_, err = Run("j", r)
	if got := codeOf(t, err); got != CodeGridInvalid {
		t.Fatalf("mixed cell counts code=%s", got)
	}
	// Fewer column cells than segments.
	r = layeredBaseRequest()
	r.Profile.Layers[0].NumLayers = 0
	r.Profile.Layers[1].NumLayers = 0
	r.Column.NZ = 1
	_, err = Run("j", r)
	if got := codeOf(t, err); got != CodeGridInvalid {
		t.Fatalf("too few cells code=%s", got)
	}
	// Both material and profile given.
	r = layeredBaseRequest()
	r.Material = Material{Alpha: 6, N: 2, ThetaR: 0.05, ThetaS: 0.4, Ks: 5e-5}
	_, err = Run("j", r)
	if got := codeOf(t, err); got != CodeMaterialChoice {
		t.Fatalf("material+profile conflict code=%s", got)
	}
}

// TestLayeredInitialValidationPerSegment: the initial water-content
// bounds are those of the segment owning each cell.
func TestLayeredInitialValidationPerSegment(t *testing.T) {
	// 0.045 is legal in the coarse segment (thetaR 0.04) but not in the
	// fine one (thetaR 0.05).
	r := layeredBaseRequest()
	th := repeatSlice(0.12, 60)
	for i := 24; i < 60; i++ {
		th[i] = 0.045
	}
	r.Initial.WaterContent = th
	if _, err := Run("ok", r); err != nil {
		t.Fatalf("per-segment legal initial profile rejected: %v", err)
	}
	th[10] = 0.045 // illegal in the fine segment
	r.Initial.WaterContent = th
	_, err := Run("bad", r)
	if got := codeOf(t, err); got != CodeProfileInvalid {
		t.Fatalf("code=%s want %s", got, CodeProfileInvalid)
	}
	var ve *ValidationError
	_ = errors.As(err, &ve)
	if !strings.Contains(ve.Message, "layer 10") {
		t.Fatalf("error must name the offending layer: %v", ve.Message)
	}
}

// TestLayeredHydrostaticInitialJob: hydrostatic initialisation through
// the job path uses each segment's own curve and stays motionless.
func TestLayeredHydrostaticInitialJob(t *testing.T) {
	r := layeredBaseRequest()
	r.Initial = Initial{Kind: "hydrostatic", WaterTableDepthM: 1.5}
	r.Boundary = Boundary{Top: solver.TopZeroFlux, Bottom: solver.BottomZeroFlux}
	r.Time = TimeSpec{TotalTime: 3600, StepSize: 300}
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
	// Water content jumps across the interface (different curves).
	if math.Abs(first.Layers[23].WaterContent-first.Layers[24].WaterContent) < 1e-3 {
		t.Fatal("hydrostatic theta should jump at the interface")
	}
	if math.Abs(res.TotalMassResidualM) > 1e-10 {
		t.Fatalf("closure %g", res.TotalMassResidualM)
	}
}

// TestLayeredStepAndFullRunAgree: the single-step endpoint and the full
// run share the same layered machinery and must agree from the same
// initial state.
func TestLayeredStepAndFullRunAgree(t *testing.T) {
	r := layeredBaseRequest()
	full, err := Run("full", r)
	if err != nil {
		t.Fatal(err)
	}
	one, err := RunStep("one", StepRequest{
		Column:   r.Column,
		Profile:  r.Profile,
		Initial:  r.Initial,
		Boundary: r.Boundary,
		StepSize: r.Time.StepSize,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !one.MaterialConfig.Layered || len(one.MaterialConfig.Segments) != 2 {
		t.Fatal("single-step response must echo the layered profile")
	}
	for i := range one.After {
		a, b := full.Steps[0].Layers[i], one.After[i]
		if math.Abs(a.WaterContent-b.WaterContent) > 1e-13 ||
			math.Abs(a.PressureHead-b.PressureHead) > 1e-12 {
			t.Fatalf("layer %d differs between paths", i)
		}
	}
	if math.Abs(full.Steps[0].MassBalanceResidM-one.Step.MassBalanceResidM) > 1e-14 {
		t.Fatal("closure differs between paths")
	}
}

// TestLayeredProportionalCellAllocation: without per-segment counts the
// column's cells are distributed proportionally, interfaces stay on
// faces, and every segment gets at least one cell.
func TestLayeredProportionalCellAllocation(t *testing.T) {
	r := layeredBaseRequest()
	r.Profile.Layers[0].NumLayers = 0
	r.Profile.Layers[1].NumLayers = 0
	res, err := Run("p", r)
	if err != nil {
		t.Fatal(err)
	}
	if res.Grid.NZ != 60 {
		t.Fatalf("NZ=%d want 60", res.Grid.NZ)
	}
	segs := res.MaterialConfig.Segments
	if segs[0].NumLayers+segs[1].NumLayers != 60 {
		t.Fatalf("allocated cells %d + %d != 60", segs[0].NumLayers, segs[1].NumLayers)
	}
	if segs[0].NumLayers < 1 || segs[1].NumLayers < 1 {
		t.Fatal("every segment must get at least one cell")
	}
	// 0.4/1.0 of 60 = 24 cells in segment 0.
	if segs[0].NumLayers != 24 {
		t.Fatalf("segment 0 got %d cells, want 24", segs[0].NumLayers)
	}
	if got := res.Grid.FaceDepthsM[segs[0].NumLayers]; got != 0.4 {
		t.Fatalf("interface face at %v, want exactly 0.4", got)
	}
}

// TestLayeredInterfaceFluxesExposed: the reported face fluxes let the
// caller verify per-segment balances directly.
func TestLayeredInterfaceFluxesExposed(t *testing.T) {
	res, err := Run("f", layeredBaseRequest())
	if err != nil {
		t.Fatal(err)
	}
	st := res.Steps[len(res.Steps)-1]
	if len(st.FaceFluxesM_S) != 61 {
		t.Fatalf("face fluxes %d, want 61", len(st.FaceFluxesM_S))
	}
	// Interface face flux is the single shared value across segments.
	iface := res.Grid.InterfaceFaces[0]
	q := st.FaceFluxesM_S[iface]
	if math.IsNaN(q) || math.IsInf(q, 0) {
		t.Fatal("interface flux not finite")
	}
	// Per-segment balance over the whole run from the reported fluxes is
	// exercised in the solver tests; here just confirm the column balance.
	if math.Abs(res.TotalMassResidualM) > 1e-9 {
		t.Fatalf("closure %g", res.TotalMassResidualM)
	}
}

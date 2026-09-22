package job

import (
	"errors"
	"math"
	"strings"
	"sync"
	"testing"

	"richards-service/internal/solver"
)

func baseRequest() Request {
	nz := 40
	th := make([]float64, nz)
	for i := range th {
		th[i] = 0.15
	}
	m := Material{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	return Request{
		Column:   Column{Thickness: 1.0, NZ: nz},
		Material: &m,
		Initial:  Initial{Kind: "water_content", WaterContent: th},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 5400, StepSize: 30},
	}
}

func codeOf(t *testing.T, err error) string {
	t.Helper()
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected ValidationError, got %T %v", err, err)
	}
	return ve.Code
}

func TestFiveIllegalParameterClasses(t *testing.T) {
	cases := []struct {
		name string
		mut  func(r *Request)
		code string
	}{
		{"n <= 1", func(r *Request) { r.Material.N = 1.0 }, CodeNInvalid},
		{"n < 1", func(r *Request) { r.Material.N = 0.8 }, CodeNInvalid},
		{"alpha <= 0", func(r *Request) { r.Material.Alpha = 0 }, CodeAlphaNonPositive},
		{"thetaR >= thetaS", func(r *Request) { r.Material.ThetaR = 0.40 }, CodeThetaRangeInvalid},
		{"ks <= 0", func(r *Request) { r.Material.Ks = -1e-5 }, CodeKsNonPositive},
		{"thickness zero", func(r *Request) { r.Column.Thickness = 0 }, CodeThicknessZero},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := baseRequest()
			tc.mut(&r)
			_, err := Run("j", r)
			if err == nil {
				t.Fatal("expected validation error, got nil")
			}
			if got := codeOf(t, err); got != tc.code {
				t.Fatalf("code=%s want %s", got, tc.code)
			}
		})
	}
}

func TestIllegalBoundaryAndProfile(t *testing.T) {
	r := baseRequest()
	r.Boundary.Top = "rainfall"
	_, err := Run("j", r)
	if got := codeOf(t, err); got != CodeBoundaryInvalid {
		t.Fatalf("top code=%s", got)
	}

	r = baseRequest()
	r.Boundary.Bottom = "seepage"
	_, err = Run("j", r)
	if got := codeOf(t, err); got != CodeBoundaryInvalid {
		t.Fatalf("bottom code=%s", got)
	}

	r = baseRequest()
	r.Initial.WaterContent = r.Initial.WaterContent[:10]
	_, err = Run("j", r)
	if got := codeOf(t, err); got != CodeProfileInvalid {
		t.Fatalf("profile length code=%s", got)
	}

	r = baseRequest()
	r.Initial.WaterContent[5] = r.Material.ThetaS + 0.02
	_, err = Run("j", r)
	if got := codeOf(t, err); got != CodeProfileInvalid {
		t.Fatalf("profile bounds code=%s", got)
	}
}

func TestFullRunMassClosureAndProfiles(t *testing.T) {
	mgr := NewManager()
	id, res, err := mgr.RunFull(baseRequest())
	if err != nil {
		t.Fatal(err)
	}
	if id == "" {
		t.Fatal("missing job id")
	}
	if len(res.Steps) != 180 {
		t.Fatalf("steps=%d want 180", len(res.Steps))
	}
	// Every layer stays in range; every reporting interval closes.
	for k, st := range res.Steps {
		if math.Abs(st.MassBalanceResidM) > 1e-9 {
			t.Fatalf("interval %d closure %g", k, st.MassBalanceResidM)
		}
		for _, l := range st.Layers {
			if l.WaterContent < 0.05-1e-12 || l.WaterContent > 0.40+1e-12 {
				t.Fatalf("layer out of range: %g", l.WaterContent)
			}
		}
	}
	if math.Abs(res.TotalMassResidualM) > 1e-9 {
		t.Fatalf("total closure %g", res.TotalMassResidualM)
	}
	// Infiltration: storage rises, cumulative top inflow positive.
	if !(res.FinalStorageM > res.InitialStorageM) {
		t.Fatal("storage should rise")
	}
	if res.CumTopFluxM <= 0 {
		t.Fatal("cumulative top inflow should be positive")
	}
	// Front moves downward.
	front := func(st StepOutput) float64 {
		d := -1.0
		for _, l := range st.Layers {
			if l.WaterContent > 0.15+0.3*(0.40-0.15) {
				d = l.DepthM
			}
		}
		return d
	}
	fEarly, fLate := front(res.Steps[50]), front(res.Steps[len(res.Steps)-1])
	if !(fLate > fEarly && fEarly >= 0) {
		t.Fatalf("front should advance: early=%v late=%v", fEarly, fLate)
	}
}

func TestSingleStepAndFullRunSameInitialResult(t *testing.T) {
	r := baseRequest()
	// Full run, keep only the first interval.
	full, err := Run("full", r)
	if err != nil {
		t.Fatal(err)
	}
	stepReq := StepRequest{
		Column:   r.Column,
		Material: r.Material,
		Initial:  r.Initial,
		Boundary: r.Boundary,
		StepSize: r.Time.StepSize,
	}
	one, err := RunStep("one", stepReq)
	if err != nil {
		t.Fatal(err)
	}
	if len(full.Steps[0].Layers) != len(one.After) {
		t.Fatal("layer count differs")
	}
	for i := range one.After {
		a, b := full.Steps[0].Layers[i], one.After[i]
		if math.Abs(a.WaterContent-b.WaterContent) > 1e-13 ||
			math.Abs(a.PressureHead-b.PressureHead) > 1e-12 {
			t.Fatalf("layer %d differs between paths: full th=%g h=%g ; step th=%g h=%g",
				i, a.WaterContent, a.PressureHead, b.WaterContent, b.PressureHead)
		}
	}
	if math.Abs(full.Steps[0].MassBalanceResidM-one.Step.MassBalanceResidM) > 1e-14 {
		t.Fatal("single-step closure differs from full-run first interval")
	}
}

func TestSingleStepReportsBeforeAfterAndResidual(t *testing.T) {
	m := Material{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	r := StepRequest{
		Column:   Column{Thickness: 1.0, NZ: 20},
		Material: &m,
		Initial:  Initial{Kind: "water_content", WaterContent: repeatSlice(0.15, 20)},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		StepSize: 30,
	}
	res, err := RunStep("s", r)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Before) != 20 || len(res.After) != 20 {
		t.Fatal("before/after must cover all layers")
	}
	// Before profile equals the submitted initial content.
	for _, l := range res.Before {
		if math.Abs(l.WaterContent-0.15) > 1e-12 {
			t.Fatalf("before profile not initial: %g", l.WaterContent)
		}
	}
	if math.Abs(res.Step.MassBalanceResidM) > 1e-10 {
		t.Fatalf("step closure %g", res.Step.MassBalanceResidM)
	}
}

func TestHydrostaticZeroFluxStaysStill(t *testing.T) {
	m := Material{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	r := Request{
		Column:   Column{Thickness: 1.0, NZ: 30},
		Material: &m,
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
		t.Fatalf("closure %g", res.TotalMassResidualM)
	}
	if math.Abs(res.CumTopFluxM)+math.Abs(res.CumBottomFluxM) > 0 {
		t.Fatal("zero-flux boundaries must integrate to zero")
	}
}

func TestSandExampleIsValidAndAdvances(t *testing.T) {
	req := SandPondingRequest()
	if got := ListExamples()[0].ID; got != SandPondingExampleID {
		t.Fatalf("example id %s", got)
	}
	res, err := Run(SandPondingExampleID, req)
	if err != nil {
		t.Fatal(err)
	}
	if res.FinalStorageM <= res.InitialStorageM {
		t.Fatal("preset example must gain water")
	}
	if math.Abs(res.TotalMassResidualM) > 1e-9 {
		t.Fatalf("example closure %g", res.TotalMassResidualM)
	}
}

func TestConcurrentJobsDoNotIntermix(t *testing.T) {
	mgr := NewManager()
	const n = 24
	var wg sync.WaitGroup
	results := make([]*Result, n)
	errs := make([]error, n)
	for j := 0; j < n; j++ {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			r := baseRequest()
			// Alternate distinct configurations.
			if j%2 == 1 {
				r.Material.Ks = 5e-6
				r.Time.TotalTime = 300
			}
			id, res, err := mgr.RunFull(r)
			if id == "" {
				errs[j] = errors.New("empty id")
				return
			}
			results[j], errs[j] = res, err
		}(j)
	}
	wg.Wait()
	// Reference values computed independently in the main goroutine.
	refA := mustRun(t, baseRequest())
	rSlow := baseRequest()
	rSlow.Material.Ks = 5e-6
	rSlow.Time.TotalTime = 300
	refB := mustRun(t, rSlow)

	for j := 0; j < n; j++ {
		if errs[j] != nil {
			t.Fatalf("job %d: %v", j, errs[j])
		}
		ref := refA
		if j%2 == 1 {
			ref = refB
		}
		if len(results[j].Steps) != len(ref.Steps) {
			t.Fatalf("job %d step count %d want %d", j, len(results[j].Steps), len(ref.Steps))
		}
		if math.Abs(results[j].FinalStorageM-ref.FinalStorageM) > 1e-12 {
			t.Fatalf("job %d storage %v want %v (intermixed?)",
				j, results[j].FinalStorageM, ref.FinalStorageM)
		}
		if math.Abs(results[j].CumTopFluxM-ref.CumTopFluxM) > 1e-12 {
			t.Fatalf("job %d cumulative flux %v want %v", j,
				results[j].CumTopFluxM, ref.CumTopFluxM)
		}
	}
}

func TestManagerCounters(t *testing.T) {
	mgr := NewManager()
	if _, _, err := mgr.RunFull(baseRequest()); err != nil {
		t.Fatal(err)
	}
	bad := baseRequest()
	bad.Material.N = 0.5
	if _, _, err := mgr.RunFull(bad); err == nil {
		t.Fatal("expected failure")
	}
	st := mgr.StatsSnapshot()
	if st.Completed != 1 || st.Failed != 1 || st.Submitted != 2 {
		t.Fatalf("counters wrong: %+v", st)
	}
}

func mustRun(t *testing.T, r Request) *Result {
	t.Helper()
	res, err := Run("ref", r)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func repeatSlice(v float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func TestMLockValue(t *testing.T) {
	r := baseRequest()
	res, err := Run("m", r)
	if err != nil {
		t.Fatal(err)
	}
	want := 1.0 - 1.0/2.0
	if math.Abs(res.MLocked-want) > 1e-15 {
		t.Fatalf("m=%g want %g", res.MLocked, want)
	}
}

func TestInvalidTimeSpec(t *testing.T) {
	r := baseRequest()
	r.Time = TimeSpec{TotalTime: 100, StepSize: 30}
	_, err := Run("t", r)
	if err == nil || !strings.Contains(err.Error(), "integer multiple") {
		t.Fatalf("expected non-divisible time error, got %v", err)
	}
}

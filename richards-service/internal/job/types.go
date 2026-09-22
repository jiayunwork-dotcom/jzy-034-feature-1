// Package job orchestrates "one infiltration run": validating the submitted
// description, building the constitutive model and finite-volume grid,
// marching the solver, and assembling the returned time series.
//
// A job is a self-contained value: each run gets its own solver instance and
// its own slices; runs never share mutable intermediate state.
package job

import (
	"fmt"
	"math"

	"richards-service/internal/constitutive"
	"richards-service/internal/solver"
)

// ---- request / response wire types --------------------------------------

// Request describes one infiltration run.
type Request struct {
	Column   Column          `json:"column"`
	Material Material        `json:"material"`
	Initial  Initial         `json:"initial"`
	Boundary Boundary        `json:"boundary"`
	Time     TimeSpec        `json:"time"`
	Options  *solver.Options `json:"options,omitempty"`
}

// Column describes the vertical soil column.
type Column struct {
	Thickness float64 `json:"thickness_m"` // total column thickness L [m], L > 0
	NZ        int     `json:"num_layers"`  // number of equal-thickness cells
}

// Material holds the van Genuchten–Mualem parameters.
type Material struct {
	Alpha  float64 `json:"alpha"`   // inverse air-entry head [1/m]
	N      float64 `json:"n"`       // pore-size index, n > 1
	ThetaR float64 `json:"theta_r"` // residual water content
	ThetaS float64 `json:"theta_s"` // saturated water content
	Ks     float64 `json:"ks"`      // saturated conductivity [m/s]
}

// Initial describes the initial profile. Kind "water_content" carries one
// theta per layer; "pressure_head" carries heads; "hydrostatic" builds an
// equilibrium profile with the water table at depth WaterTableDepth_m
// (z = water-table elevation; below it h >= 0).
type Initial struct {
	Kind             string    `json:"kind"`
	WaterContent     []float64 `json:"water_content,omitempty"`
	PressureHead     []float64 `json:"pressure_head,omitempty"`
	WaterTableDepthM float64   `json:"water_table_depth_m,omitempty"`
}

// Boundary describes the top and bottom boundary conditions.
type Boundary struct {
	Top        string  `json:"top"`           // "ponded_head" | "zero_flux"
	PondedHead float64 `json:"ponded_head_m"` // used when top == ponded_head
	Bottom     string  `json:"bottom"`        // "free_drainage" | "zero_flux"
}

// TimeSpec gives total simulated time and the fixed step size.
type TimeSpec struct {
	TotalTime float64 `json:"total_time_s"`
	StepSize  float64 `json:"step_size_s"`
}

// LayerSnapshot is one layer's state at one output time.
type LayerSnapshot struct {
	DepthM       float64 `json:"depth_m"`
	WaterContent float64 `json:"water_content"`
	PressureHead float64 `json:"pressure_head_m"`
}

// StepOutput is one accepted reporting interval.
type StepOutput struct {
	Index             int             `json:"index"`
	TimeS             float64         `json:"time_s"`
	TopFlux           float64         `json:"top_flux_m_s"`
	BottomFlux        float64         `json:"bottom_flux_m_s"`
	CumTopFluxM       float64         `json:"cum_top_flux_m"`
	CumBottomFluxM    float64         `json:"cum_bottom_flux_m"`
	StorageM          float64         `json:"storage_m"`
	MassBalanceResidM float64         `json:"mass_balance_residual_m"`
	MassBalanceRel    float64         `json:"mass_balance_relative"`
	Iterations        int             `json:"iterations"`
	Substeps          int             `json:"substeps"`
	Layers            []LayerSnapshot `json:"layers"`
}

// Result is the full-run result.
type Result struct {
	JobID              string       `json:"job_id"`
	Grid               GridInfo     `json:"grid"`
	MLocked            float64      `json:"m_locked"`
	InitialStorageM    float64      `json:"initial_storage_m"`
	FinalStorageM      float64      `json:"final_storage_m"`
	CumTopFluxM        float64      `json:"cum_top_flux_m"`
	CumBottomFluxM     float64      `json:"cum_bottom_flux_m"`
	TotalMassResidualM float64      `json:"total_mass_balance_residual_m"`
	Steps              []StepOutput `json:"steps"`
}

// GridInfo echoes the discretisation.
type GridInfo struct {
	NZ             int       `json:"num_layers"`
	CellThicknessM float64   `json:"cell_thickness_m"`
	DepthM         []float64 `json:"node_depths_m"`
}

// StepRequest advances a single time step only.
type StepRequest struct {
	Column   Column          `json:"column"`
	Material Material        `json:"material"`
	Initial  Initial         `json:"initial"`
	Boundary Boundary        `json:"boundary"`
	StepSize float64         `json:"step_size_s"`
	Options  *solver.Options `json:"options,omitempty"`
}

// StepResultResponse returns before/after profiles plus the balance residual
// for upstream step-by-step checking.
type StepResultResponse struct {
	JobID   string          `json:"job_id"`
	Grid    GridInfo        `json:"grid"`
	MLocked float64         `json:"m_locked"`
	Step    StepOutput      `json:"step"`
	Before  []LayerSnapshot `json:"before"`
	After   []LayerSnapshot `json:"after"`
}

// ---- structured validation errors --------------------------------------

// ValidationError is returned before any time marching starts.
type ValidationError struct {
	Code    string `json:"code"`
	Field   string `json:"field"`
	Message string `json:"message"`
}

// Validation error codes (one per illegal-parameter class required).
const (
	CodeNInvalid          = "N_NOT_GREATER_THAN_ONE"
	CodeAlphaNonPositive  = "ALPHA_NON_POSITIVE"
	CodeThetaRangeInvalid = "THETA_R_GE_THETA_S"
	CodeKsNonPositive     = "KS_NON_POSITIVE"
	CodeThicknessZero     = "COLUMN_THICKNESS_NON_POSITIVE"
	CodeProfileInvalid    = "INITIAL_PROFILE_INVALID"
	CodeBoundaryInvalid   = "BOUNDARY_INVALID"
	CodeTimeInvalid       = "TIME_SPEC_INVALID"
	CodeGridInvalid       = "GRID_INVALID"
)

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s (%s): %s", e.Code, e.Field, e.Message)
}

func validate(m Material, c Column) *ValidationError {
	if !isFinite(m.N) || m.N <= 1 {
		return &ValidationError{Code: CodeNInvalid, Field: "material.n",
			Message: "n must be > 1; m is locked to 1-1/n"}
	}
	if !isFinite(m.Alpha) || m.Alpha <= 0 {
		return &ValidationError{Code: CodeAlphaNonPositive, Field: "material.alpha",
			Message: "alpha (inverse air-entry head) must be > 0"}
	}
	if !isFinite(m.ThetaR) || !isFinite(m.ThetaS) || m.ThetaR >= m.ThetaS {
		return &ValidationError{Code: CodeThetaRangeInvalid,
			Field:   "material.theta_r/theta_s",
			Message: "need 0 <= theta_r < theta_s"}
	}
	if !isFinite(m.Ks) || m.Ks <= 0 {
		return &ValidationError{Code: CodeKsNonPositive, Field: "material.ks",
			Message: "saturated conductivity ks must be > 0"}
	}
	if !isFinite(c.Thickness) || c.Thickness <= 0 {
		return &ValidationError{Code: CodeThicknessZero, Field: "column.thickness_m",
			Message: "column thickness must be > 0"}
	}
	if c.NZ < 1 {
		return &ValidationError{Code: CodeGridInvalid, Field: "column.num_layers",
			Message: "num_layers must be >= 1"}
	}
	return nil
}

func validateBoundary(b Boundary) *ValidationError {
	switch b.Top {
	case solver.TopPondedHead:
		if !isFinite(b.PondedHead) || b.PondedHead < 0 {
			return &ValidationError{Code: CodeBoundaryInvalid, Field: "boundary.ponded_head_m",
				Message: "ponded head must be >= 0"}
		}
	case solver.TopZeroFlux:
	default:
		return &ValidationError{Code: CodeBoundaryInvalid, Field: "boundary.top",
			Message: "top must be one of: ponded_head, zero_flux"}
	}
	switch b.Bottom {
	case solver.BottomFreeDrainage, solver.BottomZeroFlux:
	default:
		return &ValidationError{Code: CodeBoundaryInvalid, Field: "boundary.bottom",
			Message: "bottom must be one of: free_drainage, zero_flux"}
	}
	return nil
}

func validateInitial(in Initial, nz int, p constitutive.Params) *ValidationError {
	switch in.Kind {
	case "water_content":
		if len(in.WaterContent) != nz {
			return &ValidationError{Code: CodeProfileInvalid, Field: "initial.water_content",
				Message: fmt.Sprintf("need %d values, got %d", nz, len(in.WaterContent))}
		}
		for i, th := range in.WaterContent {
			if !isFinite(th) || th < p.ThetaR-1e-12 || th > p.ThetaS+1e-12 {
				return &ValidationError{Code: CodeProfileInvalid, Field: "initial.water_content",
					Message: fmt.Sprintf("layer %d value %g outside [theta_r=%g, theta_s=%g]",
						i, th, p.ThetaR, p.ThetaS)}
			}
		}
	case "pressure_head":
		if len(in.PressureHead) != nz {
			return &ValidationError{Code: CodeProfileInvalid, Field: "initial.pressure_head",
				Message: fmt.Sprintf("need %d values, got %d", nz, len(in.PressureHead))}
		}
		for i, h := range in.PressureHead {
			if !isFinite(h) {
				return &ValidationError{Code: CodeProfileInvalid, Field: "initial.pressure_head",
					Message: fmt.Sprintf("layer %d head not finite", i)}
			}
		}
	case "hydrostatic":
		if !isFinite(in.WaterTableDepthM) {
			return &ValidationError{Code: CodeProfileInvalid, Field: "initial.water_table_depth_m",
				Message: "water table depth must be finite"}
		}
	default:
		return &ValidationError{Code: CodeProfileInvalid, Field: "initial.kind",
			Message: "initial kind must be water_content, pressure_head or hydrostatic"}
	}
	return nil
}

func validateTime(total, dt float64) *ValidationError {
	if !isFinite(total) || total <= 0 {
		return &ValidationError{Code: CodeTimeInvalid, Field: "time.total_time_s",
			Message: "total_time_s must be > 0"}
	}
	if !isFinite(dt) || dt <= 0 {
		return &ValidationError{Code: CodeTimeInvalid, Field: "time.step_size_s",
			Message: "step_size_s must be > 0"}
	}
	if dt > total {
		return &ValidationError{Code: CodeTimeInvalid, Field: "time.step_size_s",
			Message: "step_size_s must not exceed total_time_s"}
	}
	return nil
}

func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// ---- execution ----------------------------------------------------------

func buildParams(m Material) constitutive.Params {
	return constitutive.Params{Alpha: m.Alpha, N: m.N,
		ThetaR: m.ThetaR, ThetaS: m.ThetaS, Ks: m.Ks}
}

func initialHeads(in Initial, g solver.Grid, p constitutive.Params) ([]float64, error) {
	switch in.Kind {
	case "pressure_head":
		return append([]float64(nil), in.PressureHead...), nil
	case "water_content":
		h := make([]float64, g.NZ)
		for i, th := range in.WaterContent {
			h[i] = p.HeadFromWaterContent(th)
		}
		return h, nil
	case "hydrostatic":
		h := make([]float64, g.NZ)
		for i := 0; i < g.NZ; i++ {
			// constant total head: H = h - z = -z_wt, hence h = z - z_wt
			h[i] = g.Z[i] - in.WaterTableDepthM
		}
		return h, nil
	}
	return nil, fmt.Errorf("unknown initial kind")
}

// buildSolver centralises solver construction so the single-step and
// full-run paths cannot drift apart.
func buildSolver(req Request) (*solver.Solver, solver.Grid, *ValidationError) {
	if e := validate(req.Material, req.Column); e != nil {
		return nil, solver.Grid{}, e
	}
	if e := validateBoundary(req.Boundary); e != nil {
		return nil, solver.Grid{}, e
	}
	p := buildParams(req.Material)
	g := solver.NewGrid(req.Column.NZ, req.Column.Thickness)
	if e := validateInitial(req.Initial, g.NZ, p); e != nil {
		return nil, solver.Grid{}, e
	}
	heads, err := initialHeads(req.Initial, g, p)
	if err != nil {
		return nil, solver.Grid{}, &ValidationError{Code: CodeProfileInvalid,
			Field: "initial", Message: err.Error()}
	}
	opts := solver.DefaultOptions()
	if req.Options != nil {
		opts = mergeOptions(opts, *req.Options)
	}
	s, err := solver.NewSolver(p, g, req.Boundary.Top, req.Boundary.PondedHead,
		req.Boundary.Bottom, heads, opts)
	if err != nil {
		return nil, solver.Grid{}, &ValidationError{Code: CodeProfileInvalid,
			Field: "initial", Message: err.Error()}
	}
	return s, g, nil
}

func mergeOptions(base, override solver.Options) solver.Options {
	o := base
	if override.MaxIterations > 0 {
		o.MaxIterations = override.MaxIterations
	}
	if override.HeadTol > 0 {
		o.HeadTol = override.HeadTol
	}
	if override.ResidualTol > 0 {
		o.ResidualTol = override.ResidualTol
	}
	if override.Relaxation > 0 && override.Relaxation <= 1 {
		o.Relaxation = override.Relaxation
	}
	return o
}

func snapshots(g solver.Grid, h, theta []float64) []LayerSnapshot {
	out := make([]LayerSnapshot, g.NZ)
	for i := 0; i < g.NZ; i++ {
		out[i] = LayerSnapshot{DepthM: g.Z[i], WaterContent: theta[i],
			PressureHead: h[i]}
	}
	return out
}

func gridInfo(g solver.Grid) GridInfo {
	return GridInfo{NZ: g.NZ, CellThicknessM: g.Dz,
		DepthM: append([]float64(nil), g.Z...)}
}

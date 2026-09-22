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
//
// Exactly one of Material / Materials must be given:
//   - Material: the legacy homogeneous column (one parameter set for the
//     whole column);
//   - Materials: a layered profile, segments listed top to bottom, each with
//     its own thickness, van Genuchten–Mualem parameters and (optionally) its
//     own number of equal-thickness cells. Segment interfaces land exactly
//     on grid faces.
type Request struct {
	Column    Column          `json:"column"`
	Material  *Material       `json:"material,omitempty"`
	Materials []MaterialLayer `json:"materials,omitempty"`
	Initial   Initial         `json:"initial"`
	Boundary  Boundary        `json:"boundary"`
	Time      TimeSpec        `json:"time"`
	Options   *solver.Options `json:"options,omitempty"`
}

// Column describes the vertical soil column.
type Column struct {
	Thickness float64 `json:"thickness_m"` // total column thickness L [m], L > 0
	NZ        int     `json:"num_layers"`  // number of equal-thickness cells (homogeneous column)
}

// Material holds the van Genuchten–Mualem parameters.
type Material struct {
	Alpha  float64 `json:"alpha"`   // inverse air-entry head [1/m]
	N      float64 `json:"n"`       // pore-size index, n > 1
	ThetaR float64 `json:"theta_r"` // residual water content
	ThetaS float64 `json:"theta_s"` // saturated water content
	Ks     float64 `json:"ks"`      // saturated conductivity [m/s]
}

// MaterialLayer is one segment of a layered soil profile. Segments butt end
// to end from top to bottom and their thicknesses must sum exactly to the
// column thickness. NumLayers is optional; when omitted cells are allocated
// proportionally to thickness so the total matches column.num_layers.
type MaterialLayer struct {
	Thickness float64 `json:"thickness_m"`          // segment thickness [m], > 0
	Alpha     float64 `json:"alpha"`                // inverse air-entry head [1/m]
	N         float64 `json:"n"`                    // pore-size index, n > 1
	ThetaR    float64 `json:"theta_r"`              // residual water content
	ThetaS    float64 `json:"theta_s"`              // saturated water content
	Ks        float64 `json:"ks"`                   // saturated conductivity [m/s]
	NumLayers int     `json:"num_layers,omitempty"` // equal-thickness cells in this segment
}

// asMaterial views the layer's five constitutive parameters.
func (l MaterialLayer) asMaterial() Material {
	return Material{Alpha: l.Alpha, N: l.N, ThetaR: l.ThetaR, ThetaS: l.ThetaS, Ks: l.Ks}
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
	// Segment is the 0-based material segment this cell belongs to; absent in
	// a homogeneous column it is 0 (the single material).
	Segment int `json:"segment"`
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

// MaterialInfo echoes one segment of the resolved material profile actually
// used by the job: its thickness, five van Genuchten–Mualem parameters and
// the cell index range [CellStart, CellEnd) that belongs to it.
type MaterialInfo struct {
	Index     int     `json:"index"`
	Thickness float64 `json:"thickness_m"`
	Alpha     float64 `json:"alpha"`
	N         float64 `json:"n"`
	ThetaR    float64 `json:"theta_r"`
	ThetaS    float64 `json:"theta_s"`
	Ks        float64 `json:"ks"`
	CellStart int     `json:"cell_start"`
	CellEnd   int     `json:"cell_end"`
}

// Result is the full-run result.
type Result struct {
	JobID              string         `json:"job_id"`
	Grid               GridInfo       `json:"grid"`
	MLocked            float64        `json:"m_locked"`
	Materials          []MaterialInfo `json:"materials"`
	InitialStorageM    float64        `json:"initial_storage_m"`
	FinalStorageM      float64        `json:"final_storage_m"`
	CumTopFluxM        float64        `json:"cum_top_flux_m"`
	CumBottomFluxM     float64        `json:"cum_bottom_flux_m"`
	TotalMassResidualM float64        `json:"total_mass_balance_residual_m"`
	Steps              []StepOutput   `json:"steps"`
}

// GridInfo echoes the discretisation.
type GridInfo struct {
	NZ             int       `json:"num_layers"`
	CellThicknessM float64   `json:"cell_thickness_m"`
	DepthM         []float64 `json:"node_depths_m"`
	// CellThicknessMAll is the per-cell thickness [m]; length NZ. It equals
	// the uniform CellThicknessM for a homogeneous column.
	CellThicknessMAll []float64 `json:"cell_thickness_m_all,omitempty"`
	// SegmentOfCell gives the material segment index of each cell.
	SegmentOfCell []int `json:"segment_of_cell,omitempty"`
	// SegmentInterfaceDepthsM are the depths of the material interfaces
	// (including top 0 and bottom depth); aligned exactly with grid faces.
	SegmentInterfaceDepthsM []float64 `json:"segment_interface_depths_m,omitempty"`
}

// StepRequest advances a single time step only. As with Request, exactly one
// of Material / Materials must be present.
type StepRequest struct {
	Column    Column          `json:"column"`
	Material  *Material       `json:"material,omitempty"`
	Materials []MaterialLayer `json:"materials,omitempty"`
	Initial   Initial         `json:"initial"`
	Boundary  Boundary        `json:"boundary"`
	StepSize  float64         `json:"step_size_s"`
	Options   *solver.Options `json:"options,omitempty"`
}

// StepResultResponse returns before/after profiles plus the balance residual
// for upstream step-by-step checking.
type StepResultResponse struct {
	JobID     string          `json:"job_id"`
	Grid      GridInfo        `json:"grid"`
	MLocked   float64         `json:"m_locked"`
	Materials []MaterialInfo  `json:"materials"`
	Step      StepOutput      `json:"step"`
	Before    []LayerSnapshot `json:"before"`
	After     []LayerSnapshot `json:"after"`
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
	CodeMaterialInvalid   = "MATERIAL_INPUT_INVALID"
)

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s (%s): %s", e.Code, e.Field, e.Message)
}

// validateMaterial checks the five constitutive/geometry parameter classes.
// fieldPrefix names the offending location (e.g. "material" or
// "materials[2]") so layered errors identify the bad segment.
func validateMaterial(m Material, fieldPrefix string) *ValidationError {
	if !isFinite(m.N) || m.N <= 1 {
		return &ValidationError{Code: CodeNInvalid, Field: fieldPrefix + ".n",
			Message: fmt.Sprintf("%s: n must be > 1; m is locked to 1-1/n", fieldPrefix)}
	}
	if !isFinite(m.Alpha) || m.Alpha <= 0 {
		return &ValidationError{Code: CodeAlphaNonPositive, Field: fieldPrefix + ".alpha",
			Message: fmt.Sprintf("%s: alpha (inverse air-entry head) must be > 0", fieldPrefix)}
	}
	if !isFinite(m.ThetaR) || !isFinite(m.ThetaS) || m.ThetaR >= m.ThetaS {
		return &ValidationError{Code: CodeThetaRangeInvalid,
			Field:   fieldPrefix + ".theta_r/theta_s",
			Message: fmt.Sprintf("%s: need 0 <= theta_r < theta_s", fieldPrefix)}
	}
	if !isFinite(m.Ks) || m.Ks <= 0 {
		return &ValidationError{Code: CodeKsNonPositive, Field: fieldPrefix + ".ks",
			Message: fmt.Sprintf("%s: saturated conductivity ks must be > 0", fieldPrefix)}
	}
	return nil
}

func validateColumn(c Column) *ValidationError {
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

// validateLayers checks a segmented material profile: at least one segment,
// positive per-segment thickness summing exactly to the column depth, valid
// per-segment parameters, and (if any segment specifies a cell count) exact
// coverage of column.num_layers.
func validateLayers(ls []MaterialLayer, c Column) *ValidationError {
	if len(ls) < 1 {
		return &ValidationError{Code: CodeMaterialInvalid, Field: "materials",
			Message: "materials must contain at least one layer"}
	}
	sum := 0.0
	explicit := 0
	anyExplicit := false
	for k := range ls {
		prefix := fmt.Sprintf("materials[%d]", k)
		if e := validateMaterial(ls[k].asMaterial(), prefix); e != nil {
			return e
		}
		if !isFinite(ls[k].Thickness) || ls[k].Thickness <= 0 {
			return &ValidationError{Code: CodeMaterialInvalid,
				Field:   fmt.Sprintf("materials[%d].thickness_m", k),
				Message: fmt.Sprintf("%s: layer thickness must be > 0", prefix)}
		}
		sum += ls[k].Thickness
		if ls[k].NumLayers > 0 {
			anyExplicit = true
			explicit += ls[k].NumLayers
		}
		if ls[k].NumLayers < 0 {
			return &ValidationError{Code: CodeMaterialInvalid,
				Field:   fmt.Sprintf("materials[%d].num_layers", k),
				Message: fmt.Sprintf("%s: num_layers must be >= 1", prefix)}
		}
	}
	tol := 1e-10 * c.Thickness
	if tol < 1e-12 {
		tol = 1e-12
	}
	if d := sum - c.Thickness; d > tol || d < -tol {
		return &ValidationError{Code: CodeMaterialInvalid, Field: "materials",
			Message: fmt.Sprintf(
				"sum of layer thicknesses %g must equal column thickness %g", sum, c.Thickness)}
	}
	if anyExplicit && explicit != c.NZ {
		return &ValidationError{Code: CodeMaterialInvalid, Field: "materials",
			Message: fmt.Sprintf(
				"sum of per-layer num_layers %d must equal column.num_layers %d", explicit, c.NZ)}
	}
	if !anyExplicit && len(ls) > c.NZ {
		return &ValidationError{Code: CodeMaterialInvalid, Field: "materials",
			Message: fmt.Sprintf(
				"cannot give every one of %d layers at least one cell with column.num_layers=%d",
				len(ls), c.NZ)}
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

// validateInitial validates the initial-profile descriptor. Per-cell water
// contents are checked against each cell's own material bounds (mats[i]).
func validateInitial(in Initial, nz int, mats []constitutive.Params) *ValidationError {
	switch in.Kind {
	case "water_content":
		if len(in.WaterContent) != nz {
			return &ValidationError{Code: CodeProfileInvalid, Field: "initial.water_content",
				Message: fmt.Sprintf("need %d values, got %d", nz, len(in.WaterContent))}
		}
		for i, th := range in.WaterContent {
			p := mats[i]
			if !isFinite(th) || th < p.ThetaR-1e-12 || th > p.ThetaS+1e-12 {
				return &ValidationError{Code: CodeProfileInvalid, Field: "initial.water_content",
					Message: fmt.Sprintf(
						"layer %d (segment material theta_r=%g theta_s=%g) value %g outside bounds",
						i, p.ThetaR, p.ThetaS, th)}
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

// resolvedProfile is the material structure actually used by a job: one
// parameter set per cell plus the ordered segment metadata. In homogeneous
// jobs Segments has length 1 and every cell uses Params[0].
type resolvedProfile struct {
	Params   []constitutive.Params // length g.NZ
	Segments []MaterialInfo        // ordered top-to-bottom
}

// segmentParams is a validated material layer plus its resolved cell range.
type segmentSpec struct {
	thick float64
	mat   Material
	nz    int
}

// resolveProfile validates and expands the wire material description into
// per-cell parameters and segment metadata, and builds the matching grid
// (segment interfaces aligned exactly with faces).
func resolveProfile(mat *Material, layers []MaterialLayer, c Column,
) (solver.Grid, resolvedProfile, *ValidationError) {
	homogeneous := mat != nil
	if homogeneous && len(layers) > 0 {
		return solver.Grid{}, resolvedProfile{}, &ValidationError{Code: CodeMaterialInvalid,
			Field:   "material/materials",
			Message: "provide either material (homogeneous column) or materials (layered profile), not both"}
	}
	if !homogeneous && len(layers) == 0 {
		return solver.Grid{}, resolvedProfile{}, &ValidationError{Code: CodeMaterialInvalid,
			Field:   "material",
			Message: "one of material or materials must be provided"}
	}
	if e := validateColumn(c); e != nil {
		return solver.Grid{}, resolvedProfile{}, e
	}

	if homogeneous {
		if e := validateMaterial(*mat, "material"); e != nil {
			return solver.Grid{}, resolvedProfile{}, e
		}
		p := buildParams(*mat)
		g := solver.NewGrid(c.NZ, c.Thickness)
		prof := resolvedProfile{
			Params: repeatParams(p, c.NZ),
			Segments: []MaterialInfo{{
				Index: 0, Thickness: c.Thickness,
				Alpha: p.Alpha, N: p.N, ThetaR: p.ThetaR, ThetaS: p.ThetaS, Ks: p.Ks,
				CellStart: 0, CellEnd: c.NZ}},
		}
		return g, prof, nil
	}

	if e := validateLayers(layers, c); e != nil {
		return solver.Grid{}, resolvedProfile{}, e
	}
	nzSeg := allocateLayerCells(layers, c.NZ)
	specs := make([]solver.LayerSpec, len(layers))
	for k := range layers {
		specs[k] = solver.LayerSpec{Thickness: layers[k].Thickness, NZ: nzSeg[k]}
	}
	g, err := solver.NewLayeredGrid(c.Thickness, specs)
	if err != nil {
		return solver.Grid{}, resolvedProfile{},
			&ValidationError{Code: CodeMaterialInvalid, Field: "materials", Message: err.Error()}
	}
	prof := resolvedProfile{Params: make([]constitutive.Params, 0, c.NZ)}
	i0 := 0
	base := 0.0
	for k := range layers {
		p := buildParams(layers[k].asMaterial())
		for j := 0; j < nzSeg[k]; j++ {
			prof.Params = append(prof.Params, p)
		}
		base += layers[k].Thickness
		prof.Segments = append(prof.Segments, MaterialInfo{
			Index: k, Thickness: layers[k].Thickness,
			Alpha: p.Alpha, N: p.N, ThetaR: p.ThetaR, ThetaS: p.ThetaS, Ks: p.Ks,
			CellStart: i0, CellEnd: i0 + nzSeg[k]})
		i0 += nzSeg[k]
	}
	return g, prof, nil
}

func repeatParams(p constitutive.Params, n int) []constitutive.Params {
	out := make([]constitutive.Params, n)
	for i := range out {
		out[i] = p
	}
	return out
}

// allocateLayerCells resolves the per-segment cell counts. When callers give
// counts explicitly they were already validated to sum to totalNZ. Otherwise
// counts are apportioned proportional to thickness using the largest-
// remainder method, guaranteeing every segment gets at least one cell (the
// caller ensured len(layers) <= totalNZ) and the counts sum to totalNZ.
func allocateLayerCells(layers []MaterialLayer, totalNZ int) []int {
	out := make([]int, len(layers))
	explicit := true
	for k := range layers {
		if layers[k].NumLayers <= 0 {
			explicit = false
			break
		}
	}
	if explicit {
		for k := range layers {
			out[k] = layers[k].NumLayers
		}
		return out
	}

	totalThick := totalThickness(layers)
	type rem struct {
		k int
		r float64
	}
	allocated := 0
	remainders := make([]rem, len(layers))
	for k := range layers {
		exact := layers[k].Thickness / totalThick * float64(totalNZ)
		floor := int(exact)
		if floor < 1 {
			floor = 1 // every segment carries at least one cell
		}
		out[k] = floor
		allocated += floor
		// fractional part against the (clamped) floor
		remainders[k] = rem{k: k, r: exact - float64(floor)}
	}
	// Hand out the remaining cells to the largest fractional parts; clamping
	// thin segments to one cell can also over-shoot, handled below.
	for allocated < totalNZ {
		best, bestR := -1, -math.MaxFloat64
		for k := range layers {
			r := remainders[k].r
			if r <= 0 {
				// raw remainders exhausted: prefer the segment with the
				// thickest resulting cells to stay close to proportional.
				r = layers[k].Thickness / float64(out[k]+1)
			}
			if r > bestR {
				bestR, best = r, k
			}
		}
		out[best]++
		allocated++
		remainders[best].r = 0
	}
	for allocated > totalNZ {
		// Can only happen when several thin segments were clamped to one
		// cell; take one back from the segment whose cells are thinnest.
		worst, smallest := -1, math.MaxFloat64
		for k := range out {
			if out[k] > 1 {
				if d := layers[k].Thickness / float64(out[k]); d < smallest {
					smallest, worst = d, k
				}
			}
		}
		if worst < 0 {
			break // validated to be unreachable
		}
		out[worst]--
		allocated--
	}
	return out
}

func totalThickness(layers []MaterialLayer) float64 {
	s := 0.0
	for k := range layers {
		s += layers[k].Thickness
	}
	return s
}

// initialHeads builds the pressure-head profile. Water-content and
// hydrostatic profiles are evaluated per cell under that cell's own material
// curve; the hydrostatic profile has h = z - z_wt (constant total head) and
// only the conversion to theta (inside the solver) is material dependent.
func initialHeads(in Initial, g solver.Grid, mats []constitutive.Params) ([]float64, error) {
	switch in.Kind {
	case "pressure_head":
		return append([]float64(nil), in.PressureHead...), nil
	case "water_content":
		h := make([]float64, g.NZ)
		for i, th := range in.WaterContent {
			h[i] = mats[i].HeadFromWaterContent(th)
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
func buildSolver(req Request) (*solver.Solver, solver.Grid, resolvedProfile, *ValidationError) {
	if e := validateBoundary(req.Boundary); e != nil {
		return nil, solver.Grid{}, resolvedProfile{}, e
	}
	g, prof, e := resolveProfile(req.Material, req.Materials, req.Column)
	if e != nil {
		return nil, solver.Grid{}, resolvedProfile{}, e
	}
	if ve := validateInitial(req.Initial, g.NZ, prof.Params); ve != nil {
		return nil, solver.Grid{}, resolvedProfile{}, ve
	}
	heads, err := initialHeads(req.Initial, g, prof.Params)
	if err != nil {
		return nil, solver.Grid{}, resolvedProfile{}, &ValidationError{Code: CodeProfileInvalid,
			Field: "initial", Message: err.Error()}
	}
	opts := solver.DefaultOptions()
	if req.Options != nil {
		opts = mergeOptions(opts, *req.Options)
	}
	var s *solver.Solver
	if req.Material != nil {
		s, err = solver.NewSolver(prof.Params[0], g, req.Boundary.Top, req.Boundary.PondedHead,
			req.Boundary.Bottom, heads, opts)
	} else {
		s, err = solver.NewLayeredSolver(prof.Params, g, req.Boundary.Top, req.Boundary.PondedHead,
			req.Boundary.Bottom, heads, opts)
	}
	if err != nil {
		return nil, solver.Grid{}, resolvedProfile{}, &ValidationError{Code: CodeProfileInvalid,
			Field: "initial", Message: err.Error()}
	}
	return s, g, prof, nil
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

func snapshots(g solver.Grid, segOf []int, h, theta []float64) []LayerSnapshot {
	out := make([]LayerSnapshot, g.NZ)
	for i := 0; i < g.NZ; i++ {
		out[i] = LayerSnapshot{DepthM: g.Z[i], WaterContent: theta[i],
			PressureHead: h[i], Segment: segOf[i]}
	}
	return out
}

// segmentOfCell returns the segment index per cell from resolved metadata.
func segmentOfCell(prof resolvedProfile, nz int) []int {
	out := make([]int, nz)
	for _, seg := range prof.Segments {
		for i := seg.CellStart; i < seg.CellEnd; i++ {
			out[i] = seg.Index
		}
	}
	return out
}

func gridInfo(g solver.Grid, prof resolvedProfile) GridInfo {
	info := GridInfo{NZ: g.NZ, CellThicknessM: g.Dz,
		DepthM: append([]float64(nil), g.Z...)}
	if len(g.DzCell) > 0 {
		info.CellThicknessMAll = append([]float64(nil), g.DzCell...)
		info.SegmentOfCell = segmentOfCell(prof, g.NZ)
		depths := make([]float64, 0, len(prof.Segments)+1)
		depths = append(depths, 0)
		base := 0.0
		for _, seg := range prof.Segments {
			base += seg.Thickness
			depths = append(depths, base)
		}
		info.SegmentInterfaceDepthsM = depths
	}
	return info
}

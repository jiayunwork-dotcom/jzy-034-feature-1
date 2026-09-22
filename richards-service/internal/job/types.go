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
// Exactly one of Material and Profile must be given: Material selects the
// legacy single-material column (the whole column shares one
// van Genuchten–Mualem parameter set); Profile selects a layered column
// whose segments each carry their own thickness and parameter set.
type Request struct {
	Column   Column              `json:"column"`
	Material Material            `json:"material"`
	Profile  *SoilProfileRequest `json:"profile,omitempty"`
	Initial  Initial             `json:"initial"`
	Boundary Boundary            `json:"boundary"`
	Time     TimeSpec            `json:"time"`
	Options  *solver.Options     `json:"options,omitempty"`
}

// Column describes the vertical soil column.
type Column struct {
	Thickness float64 `json:"thickness_m"` // total column thickness L [m], L > 0
	NZ        int     `json:"num_layers"`  // number of cells (layered: total when segments omit counts)
}

// Material holds the van Genuchten–Mualem parameters.
type Material struct {
	Alpha  float64 `json:"alpha"`   // inverse air-entry head [1/m]
	N      float64 `json:"n"`       // pore-size index, n > 1
	ThetaR float64 `json:"theta_r"` // residual water content
	ThetaS float64 `json:"theta_s"` // saturated water content
	Ks     float64 `json:"ks"`      // saturated conductivity [m/s]
}

// SoilLayer is one material segment of a layered profile, listed top to
// bottom. Segments abut in depth order and their thicknesses sum to the
// column thickness. NumLayers is optional: when omitted on every segment
// the column's num_layers is distributed proportionally to thickness; when
// given on any segment it must be given on every segment.
type SoilLayer struct {
	Thickness float64 `json:"thickness_m"`
	NumLayers int     `json:"num_layers,omitempty"`
	Material
}

// SoilProfileRequest is the ordered list of material segments.
type SoilProfileRequest struct {
	Layers []SoilLayer `json:"layers"`
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
	FaceFluxesM_S     []float64       `json:"face_fluxes_m_s,omitempty"` // all NZ+1 face fluxes
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
	JobID              string         `json:"job_id"`
	Grid               GridInfo       `json:"grid"`
	MaterialConfig     MaterialConfig `json:"material_config"`
	MLocked            float64        `json:"m_locked"`
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
	CellThicknessM float64   `json:"cell_thickness_m"` // 0 for a piecewise-uniform layered grid
	Layered        bool      `json:"layered"`
	NodeDepthsM    []float64 `json:"node_depths_m"`
	FaceDepthsM    []float64 `json:"face_depths_m"`
	CellThickness  []float64 `json:"cell_thickness_m_per_layer,omitempty"`
	// SegmentOf[i] is the material-segment index owning output layer i.
	SegmentOf []int `json:"segment_of,omitempty"`
	// InterfaceFaces are the faces coinciding with material interfaces
	// (excluding the top/bottom column ends).
	InterfaceFaces []int `json:"interface_faces,omitempty"`
}

// StepRequest advances a single time step only. As for Request, exactly
// one of Material and Profile must be given.
type StepRequest struct {
	Column   Column              `json:"column"`
	Material Material            `json:"material"`
	Profile  *SoilProfileRequest `json:"profile,omitempty"`
	Initial  Initial             `json:"initial"`
	Boundary Boundary            `json:"boundary"`
	StepSize float64             `json:"step_size_s"`
	Options  *solver.Options     `json:"options,omitempty"`
}

// SegmentInfo echoes one material segment of the profile actually used by
// a job.
type SegmentInfo struct {
	Index      int     `json:"index"`
	ThicknessM float64 `json:"thickness_m"`
	NumLayers  int     `json:"num_layers"`
	Alpha      float64 `json:"alpha"`
	N          float64 `json:"n"`
	MLocked    float64 `json:"m_locked"`
	ThetaR     float64 `json:"theta_r"`
	ThetaS     float64 `json:"theta_s"`
	Ks         float64 `json:"ks"`
}

// MaterialConfig echoes the constitutive material actually used: a single
// Material block for a uniform column, or the per-segment segment list
// for a layered column.
type MaterialConfig struct {
	// Layered is false for the legacy single-material form; the Material
	// block then carries the one parameter set.
	Layered  bool          `json:"layered"`
	Material *Material     `json:"material,omitempty"`
	Segments []SegmentInfo `json:"segments,omitempty"`
}

// StepResultResponse returns before/after profiles plus the balance residual
// for upstream step-by-step checking.
type StepResultResponse struct {
	JobID          string          `json:"job_id"`
	Grid           GridInfo        `json:"grid"`
	MaterialConfig MaterialConfig  `json:"material_config"`
	MLocked        float64         `json:"m_locked"`
	Step           StepOutput      `json:"step"`
	Before         []LayerSnapshot `json:"before"`
	After          []LayerSnapshot `json:"after"`
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
	// Layered-profile structure violations.
	CodeLayerParams    = "LAYER_PARAMS_INVALID"
	CodeLayerThickness = "LAYER_THICKNESS_NON_POSITIVE"
	CodeProfileSum     = "PROFILE_THICKNESS_MISMATCH"
	CodeLayerCount     = "PROFILE_LAYER_COUNT_INVALID"
	CodeMaterialChoice = "MATERIAL_PROFILE_CONFLICT"
)

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s (%s): %s", e.Code, e.Field, e.Message)
}

// validateMaterial checks the five parameter legality classes for one
// material block (used for the uniform column and for each profile layer).
// fieldPrefix names the offending request field in the error.
func validateMaterial(m Material, fieldPrefix string) *ValidationError {
	if !isFinite(m.N) || m.N <= 1 {
		return &ValidationError{Code: CodeNInvalid, Field: fieldPrefix + ".n",
			Message: "n must be > 1; m is locked to 1-1/n"}
	}
	if !isFinite(m.Alpha) || m.Alpha <= 0 {
		return &ValidationError{Code: CodeAlphaNonPositive, Field: fieldPrefix + ".alpha",
			Message: "alpha (inverse air-entry head) must be > 0"}
	}
	if !isFinite(m.ThetaR) || !isFinite(m.ThetaS) || m.ThetaR >= m.ThetaS {
		return &ValidationError{Code: CodeThetaRangeInvalid,
			Field:   fieldPrefix + ".theta_r/theta_s",
			Message: "need 0 <= theta_r < theta_s"}
	}
	if !isFinite(m.Ks) || m.Ks <= 0 {
		return &ValidationError{Code: CodeKsNonPositive, Field: fieldPrefix + ".ks",
			Message: "saturated conductivity ks must be > 0"}
	}
	return nil
}

// validate checks a uniform (single-material) column.
func validate(m Material, c Column) *ValidationError {
	if e := validateMaterial(m, "material"); e != nil {
		return e
	}
	return validateColumn(c)
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

// isZeroMaterial reports whether the legacy material block carries no
// values at all (the caller selected a layered profile instead).
func isZeroMaterial(m Material) bool {
	return m == Material{}
}

// validateLayered checks the layered-profile structure itself: at least
// one layer, positive per-layer thickness, five parameter legality
// classes per layer, and the layer thicknesses summing exactly to the
// column thickness. It also resolves the per-layer cell counts.
func validateLayered(pr *SoilProfileRequest, c Column) (segCells []int, ve *ValidationError) {
	if pr == nil || len(pr.Layers) < 1 {
		return nil, &ValidationError{Code: CodeLayerCount, Field: "profile.layers",
			Message: "a layered profile needs at least one material segment"}
	}
	if e := validateColumn(c); e != nil {
		return nil, e
	}
	nl := len(pr.Layers)
	thicknessSum := 0.0
	anyCells := false
	allCells := true
	rawCells := make([]int, nl)
	for k := range pr.Layers {
		lay := &pr.Layers[k]
		pref := fmt.Sprintf("profile.layers[%d]", k)
		if e := validateMaterial(lay.Material, pref); e != nil {
			e.Message = fmt.Sprintf("segment %d: %s", k, e.Message)
			e.Code = CodeLayerParams
			e.Field = pref
			return nil, e
		}
		if !isFinite(lay.Thickness) || lay.Thickness <= 0 {
			return nil, &ValidationError{Code: CodeLayerThickness,
				Field:   pref + ".thickness_m",
				Message: fmt.Sprintf("segment %d thickness must be > 0, got %g", k, lay.Thickness)}
		}
		thicknessSum += lay.Thickness
		if lay.NumLayers < 0 {
			return nil, &ValidationError{Code: CodeGridInvalid,
				Field:   pref + ".num_layers",
				Message: fmt.Sprintf("segment %d num_layers must be >= 0 (0 = auto)", k)}
		}
		if lay.NumLayers > 0 {
			anyCells = true
		} else {
			allCells = false
		}
		rawCells[k] = lay.NumLayers
	}
	// Thickness sum must equal the column thickness exactly (tight
	// relative tolerance; no silent rescaling).
	tol := 1e-9 * math.Max(math.Abs(thicknessSum), math.Abs(c.Thickness))
	if math.Abs(thicknessSum-c.Thickness) > tol {
		return nil, &ValidationError{Code: CodeProfileSum, Field: "profile.layers",
			Message: fmt.Sprintf("segment thicknesses sum to %g but column thickness is %g",
				thicknessSum, c.Thickness)}
	}
	if anyCells && !allCells {
		return nil, &ValidationError{Code: CodeGridInvalid, Field: "profile.layers",
			Message: "num_layers must be given on every segment or on none"}
	}
	if allCells {
		tot := 0
		for _, n := range rawCells {
			tot += n
		}
		if tot != c.NZ {
			return nil, &ValidationError{Code: CodeGridInvalid, Field: "profile.layers",
				Message: fmt.Sprintf("segment num_layers sum %d must equal column.num_layers %d",
					tot, c.NZ)}
		}
		return rawCells, nil
	}
	// No per-layer counts: distribute the column's cell count
	// proportionally to thickness (largest-remainder), guaranteeing at
	// least one cell per segment and exactly column.NZ in total.
	if c.NZ < nl {
		return nil, &ValidationError{Code: CodeGridInvalid, Field: "column.num_layers",
			Message: fmt.Sprintf("num_layers %d cannot be fewer than the %d profile segments",
				c.NZ, nl)}
	}
	segCells = allocateCells(rawThicknesses(pr), c.NZ)
	return segCells, nil
}

func rawThicknesses(pr *SoilProfileRequest) []float64 {
	t := make([]float64, len(pr.Layers))
	for i := range pr.Layers {
		t[i] = pr.Layers[i].Thickness
	}
	return t
}

// allocateCells distributes nz cells over segments proportional to
// thickness using the largest-remainder rule, with one cell minimum per
// segment; the total is exactly nz.
func allocateCells(thickness []float64, nz int) []int {
	ns := len(thickness)
	if nz < ns {
		// Caller validation rejects this, but never produce a zero-cell
		// segment silently.
		nz = ns
	}
	depth := 0.0
	for _, t := range thickness {
		depth += t
	}
	cells := make([]int, ns)
	remainders := make([]float64, ns)
	assigned := 0
	for k, t := range thickness {
		exact := float64(nz) * t / depth
		floor := int(exact)
		if floor < 1 {
			floor = 1
		}
		cells[k] = floor
		remainders[k] = exact - math.Floor(exact)
		assigned += floor
	}
	// Hand out remaining cells to the largest remainders.
	for assigned < nz {
		best := -1
		bestR := -1.0
		for k := range cells {
			if remainders[k] > bestR {
				bestR, best = remainders[k], k
			}
		}
		if best < 0 {
			best = 0
		}
		cells[best]++
		remainders[best] = -1
		assigned++
	}
	// Over-allocation (extreme rounding): peel cells off the largest
	// segments beyond their one-cell floor.
	for assigned > nz {
		best := 0
		for k := range cells {
			if cells[k] > cells[best] {
				best = k
			}
		}
		if cells[best] <= 1 {
			break
		}
		cells[best]--
		assigned--
	}
	return cells
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

// bounds is one cell's water-content legality interval: that of the
// material segment owning the cell.
type bounds struct{ thetaR, thetaS float64 }

// validateInitial validates the initial profile against the per-cell
// material bounds: the same numerical water content may be legal in one
// segment and out of range in another, so the check must follow the
// cell-to-segment ownership.
func validateInitial(in Initial, nz int, cellBounds []bounds) *ValidationError {
	switch in.Kind {
	case "water_content":
		if len(in.WaterContent) != nz {
			return &ValidationError{Code: CodeProfileInvalid, Field: "initial.water_content",
				Message: fmt.Sprintf("need %d values, got %d", nz, len(in.WaterContent))}
		}
		for i, th := range in.WaterContent {
			b := cellBounds[i]
			if !isFinite(th) || th < b.thetaR-1e-12 || th > b.thetaS+1e-12 {
				return &ValidationError{Code: CodeProfileInvalid, Field: "initial.water_content",
					Message: fmt.Sprintf("layer %d value %g outside its segment range "+
						"[theta_r=%g, theta_s=%g]", i, th, b.thetaR, b.thetaS)}
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

// initialHeads converts the requested initial profile into pressure heads.
// Water-content values are inverted with the retention curve of the
// material segment owning each cell; a hydrostatic profile (constant total
// head) is material-independent in h, and each cell's water content then
// follows from its own segment's curve inside the solver.
func initialHeads(in Initial, g solver.Grid, cellP []constitutive.Params) ([]float64, error) {
	switch in.Kind {
	case "pressure_head":
		return append([]float64(nil), in.PressureHead...), nil
	case "water_content":
		h := make([]float64, g.NZ)
		for i, th := range in.WaterContent {
			h[i] = cellP[i].HeadFromWaterContent(th)
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

// builtConfig is the validated, fully-resolved solver construction shared
// by the single-step and full-run paths so they cannot drift apart.
type builtConfig struct {
	solver *solver.Solver
	grid   solver.Grid
	// materialConfig echoes the constitutive structure actually used.
	materialConfig MaterialConfig
	// mLocked is the locked m of the (first) material segment.
	mLocked float64
}

// buildSolver centralises solver construction so the single-step and
// full-run paths cannot drift apart. Exactly one of req.Material /
// req.Profile must be supplied; both paths end up in the same solver
// machinery (single-material is just the one-segment case).
func buildSolver(req Request) (*builtConfig, *ValidationError) {
	if e := validateBoundary(req.Boundary); e != nil {
		return nil, e
	}
	layered := req.Profile != nil
	if layered && !isZeroMaterial(req.Material) {
		return nil, &ValidationError{Code: CodeMaterialChoice, Field: "material/profile",
			Message: "give either material (uniform column) or profile (layered column), not both"}
	}
	if !layered {
		return buildUniformSolver(req)
	}
	return buildLayeredSolver(req)
}

// buildUniformSolver is the legacy single-material path; numerics are
// bit-identical to the pre-layering implementation.
func buildUniformSolver(req Request) (*builtConfig, *ValidationError) {
	if e := validate(req.Material, req.Column); e != nil {
		return nil, e
	}
	p := buildParams(req.Material)
	g := solver.NewGrid(req.Column.NZ, req.Column.Thickness)
	cellP := make([]constitutive.Params, g.NZ)
	cellB := make([]bounds, g.NZ)
	for i := range cellP {
		cellP[i] = p
		cellB[i] = bounds{thetaR: p.ThetaR, thetaS: p.ThetaS}
	}
	if e := validateInitial(req.Initial, g.NZ, cellB); e != nil {
		return nil, e
	}
	heads, err := initialHeads(req.Initial, g, cellP)
	if err != nil {
		return nil, &ValidationError{Code: CodeProfileInvalid,
			Field: "initial", Message: err.Error()}
	}
	opts := solver.DefaultOptions()
	if req.Options != nil {
		opts = mergeOptions(opts, *req.Options)
	}
	s, err := solver.NewSolver(p, g, req.Boundary.Top, req.Boundary.PondedHead,
		req.Boundary.Bottom, heads, opts)
	if err != nil {
		return nil, &ValidationError{Code: CodeProfileInvalid,
			Field: "initial", Message: err.Error()}
	}
	m := req.Material
	return &builtConfig{
		solver: s, grid: g,
		materialConfig: MaterialConfig{Layered: false, Material: &m},
		mLocked:        p.M(),
	}, nil
}

// buildLayeredSolver validates the layered profile, allocates cells per
// segment (interfaces land exactly on cell faces), and builds the solver
// with per-cell constitutive parameters.
func buildLayeredSolver(req Request) (*builtConfig, *ValidationError) {
	segCells, ve := validateLayered(req.Profile, req.Column)
	if ve != nil {
		return nil, ve
	}
	pr := req.Profile
	prof := &solver.Profile{Segments: make([]solver.Segment, len(pr.Layers))}
	segThickness := make([]float64, len(pr.Layers))
	for k := range pr.Layers {
		p := buildParams(pr.Layers[k].Material)
		prof.Segments[k] = solver.Segment{
			Thickness: pr.Layers[k].Thickness,
			NZ:        segCells[k],
			Params:    p,
		}
		segThickness[k] = pr.Layers[k].Thickness
	}
	g, err := solver.NewLayeredGrid(segThickness, segCells)
	if err != nil {
		return nil, &ValidationError{Code: CodeGridInvalid, Field: "profile",
			Message: err.Error()}
	}
	cellP := make([]constitutive.Params, g.NZ)
	cellB := make([]bounds, g.NZ)
	for i := 0; i < g.NZ; i++ {
		p := prof.Segments[g.MaterialIndex(i)].Params
		cellP[i] = p
		cellB[i] = bounds{thetaR: p.ThetaR, thetaS: p.ThetaS}
	}
	if e := validateInitial(req.Initial, g.NZ, cellB); e != nil {
		return nil, e
	}
	heads, err := initialHeads(req.Initial, g, cellP)
	if err != nil {
		return nil, &ValidationError{Code: CodeProfileInvalid,
			Field: "initial", Message: err.Error()}
	}
	opts := solver.DefaultOptions()
	if req.Options != nil {
		opts = mergeOptions(opts, *req.Options)
	}
	s, err := solver.NewLayeredSolver(prof, g, req.Boundary.Top, req.Boundary.PondedHead,
		req.Boundary.Bottom, heads, opts)
	if err != nil {
		return nil, &ValidationError{Code: CodeProfileInvalid,
			Field: "initial", Message: err.Error()}
	}
	segs := make([]SegmentInfo, len(prof.Segments))
	for k, seg := range prof.Segments {
		segs[k] = SegmentInfo{
			Index: k, ThicknessM: seg.Thickness, NumLayers: seg.NZ,
			Alpha: seg.Params.Alpha, N: seg.Params.N, MLocked: seg.Params.M(),
			ThetaR: seg.Params.ThetaR, ThetaS: seg.Params.ThetaS, Ks: seg.Params.Ks,
		}
	}
	return &builtConfig{
		solver: s, grid: g,
		materialConfig: MaterialConfig{Layered: true, Segments: segs},
		mLocked:        prof.Segments[0].Params.M(),
	}, nil
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
	gi := GridInfo{
		NZ:             g.NZ,
		CellThicknessM: g.Dz,
		Layered:        g.MaterialOf != nil,
		NodeDepthsM:    append([]float64(nil), g.Z...),
		FaceDepthsM:    append([]float64(nil), g.ZFace...),
	}
	if g.MaterialOf != nil {
		gi.CellThickness = append([]float64(nil), g.Dzs...)
		gi.SegmentOf = append([]int(nil), g.MaterialOf...)
		for f := 1; f < g.NZ; f++ {
			if g.MaterialOf[f-1] != g.MaterialOf[f] {
				gi.InterfaceFaces = append(gi.InterfaceFaces, f)
			}
		}
	}
	return gi
}

// Package solver advances the mixed (h-based) Richards equation on a
// one-dimensional vertical soil column:
//
//	dtheta/dt = d/dz [ K(h) * (dh/dz - 1) ],   z positive downward,
//
// discretised with a cell-centred finite-volume grid, fully implicit
// (backward Euler) time stepping and a full-Newton iteration (exact
// Jacobian of the face conductivities, backtracking line search) per time
// step.
//
// The column may be either homogeneous (one van Genuchten–Mualem parameter
// set for every cell) or a layered profile: each cell belongs to one
// material segment and uses that segment's own retention and conductivity
// curves. Material interfaces lie exactly on cell faces; no cell straddles
// two materials. Inter-cell conductivities are the two-point harmonic
// (series) conductances: within one material this is the classical harmonic
// mean, while across a material interface each side's K is evaluated with
// its own K(h) curve before being combined in series, which enforces
// continuity of the single interface flux without forcing pressure head or
// water content to match across the interface (they are independent
// unknowns and are generally different).
//
// Sign convention: boundary flux q is positive downward; qTop > 0 means
// water entering the column at the top, qBot > 0 means water leaving the
// column at the bottom.
package solver

import (
	"fmt"
	"math"

	"richards-service/internal/constitutive"
)

// Top boundary condition kinds.
const (
	TopPondedHead = "ponded_head" // prescribed pressure head hP at the surface (Dirichlet)
	TopZeroFlux   = "zero_flux"   // no-flow top
)

// Bottom boundary condition kinds.
const (
	BottomFreeDrainage = "free_drainage" // unit-gradient outflow qBot = K(h_b)
	BottomZeroFlux     = "zero_flux"     // no-flow bottom
)

// Options controls the non-linear iteration. Zero values are replaced by
// defaults in NewSolver.
type Options struct {
	// MaxIterations caps the Newton iterations of one time step.
	MaxIterations int `json:"max_iterations"`
	// HeadTol is the max-norm head increment convergence criterion [m].
	HeadTol float64 `json:"head_tol"`
	// ResidualTol is the max-norm equation residual criterion [1/s].
	ResidualTol float64 `json:"residual_tol"`
	// Relaxation is the initial line-search step factor (0,1].
	Relaxation float64 `json:"relaxation"`
}

// DefaultOptions returns the standard solver settings.
func DefaultOptions() Options {
	return Options{
		MaxIterations: 100,
		HeadTol:       1e-6,
		ResidualTol:   1e-8,
		Relaxation:    1.0,
	}
}

// capacityFloor is the artificial-compressibility floor [1/m] used only to
// keep the Newton system regular at an uncoupled saturated cell.
const capacityFloor = 1e-10

// stagnationResidual/stagnationHead accept an iterate that is effectively
// stationary: if even repeated backtracking cannot find a detectable
// reduction but the residual and update are already physically negligible,
// the step is accepted rather than reported as a false failure.
const (
	stagnationResidual = 1e-5
	stagnationHead     = 1e-6
)

// maxNewtonHeadStep [m] caps one Newton iterate's head correction; this
// trust-region bound keeps the iterate inside Newton's convergence basin
// when the front-region Jacobian is ill-conditioned.
const maxNewtonHeadStep = 0.5

// armijoC is the sufficient-decrease constant in the Armijo rule.
const armijoC = 1e-4

func maxAbs(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		if a := math.Abs(x); a > m {
			m = a
		}
	}
	return m
}

func hasNonFinite(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return true
		}
	}
	return false
}

func l2Norm(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

// faceGeom holds the fixed geometry of one interior face f between cell
// f-1 (up) and cell f (down): distances from the two cell centres to the
// face and the centre-to-centre spacing.
type faceGeom struct {
	dUp   float64 // (zFace[f] - z[f-1])
	dDown float64 // (z[f] - zFace[f])
	span  float64 // dUp + dDown
	equal bool    // centres are equally spaced (span == uniform dz bitwise)
}

// Solver is the per-job stateful time marcher. A Solver instance belongs to
// exactly one infiltration job; nothing is shared between instances.
type Solver struct {
	Params     constitutive.Params // homogeneous material (Mats == nil)
	Grid       Grid
	Opts       Options
	TopKind    string
	PondedH    float64 // surface pressure head [m] when TopKind == TopPondedHead
	BottomKind string

	// Mats[i] is the parameter set governing cell i; nil means the whole
	// column uses the single Params set. When non-nil the slice covers
	// contiguous material segments and never splits one across a face.
	Mats []constitutive.Params

	// current state
	H     []float64 // pressure heads at cell centres [m]
	Theta []float64 // volumetric water contents
	Time  float64   // accumulated simulation time [s]

	// cumulative boundary water depths [m] (flux integrated over time),
	// sign: CumTop inflow positive, CumBot outflow positive.
	CumTop float64
	CumBot float64

	steps int

	// mean selects inter-block K averaging; harmonic in production. Only the
	// equal-spacing branch honours this switch (the test suite uses it on
	// uniform columns); material interfaces always use the series
	// conductance, which is the physical requirement.
	mean interblockMean

	// precomputed per-face geometry
	faces []faceGeom
	// uniform is true when every cell has the same (bitwise) thickness;
	// the classical uniform formulas are then used operation-for-operation.
	uniform bool
	dzU     float64
}

// NewSolver builds a homogeneous-column solver from an initial pressure-head
// profile.
func NewSolver(p constitutive.Params, g Grid, topKind string, pondedH float64,
	bottomKind string, initialHeads []float64, opts Options) (*Solver, error) {
	return newSolver(p, nil, g, topKind, pondedH, bottomKind, initialHeads, opts)
}

// NewLayeredSolver builds a layered-column solver from an initial
// pressure-head profile. mats must have one parameter set per cell (use the
// grid construction helpers to expand segment lists); each cell is governed
// by its own material's retention and conductivity curves.
func NewLayeredSolver(mats []constitutive.Params, g Grid, topKind string, pondedH float64,
	bottomKind string, initialHeads []float64, opts Options) (*Solver, error) {
	return newSolver(mats[0], mats, g, topKind, pondedH, bottomKind, initialHeads, opts)
}

func newSolver(p constitutive.Params, mats []constitutive.Params, g Grid, topKind string,
	pondedH float64, bottomKind string, initialHeads []float64, opts Options) (*Solver, error) {
	if opts.MaxIterations <= 0 {
		opts = DefaultOptions()
	}
	if opts.HeadTol <= 0 {
		opts.HeadTol = DefaultOptions().HeadTol
	}
	if opts.ResidualTol <= 0 {
		opts.ResidualTol = DefaultOptions().ResidualTol
	}
	if opts.Relaxation <= 0 || opts.Relaxation > 1 {
		opts.Relaxation = 1
	}
	if len(initialHeads) != g.NZ {
		return nil, fmt.Errorf("initial profile length %d does not match grid size %d", len(initialHeads), g.NZ)
	}
	if mats != nil && len(mats) != g.NZ {
		return nil, fmt.Errorf("material profile length %d does not match grid size %d", len(mats), g.NZ)
	}
	s := &Solver{
		Params: p, Mats: mats, Grid: g, Opts: opts,
		TopKind: topKind, PondedH: pondedH, BottomKind: bottomKind,
		H:     append([]float64(nil), initialHeads...),
		Theta: make([]float64, g.NZ),
		mean:  HarmonicMean,
	}
	s.uniform = g.IsUniform()
	if s.uniform {
		// uniform columns: use the exact quotient of the legacy path
		s.dzU = g.Dz
		if len(g.DzCell) > 0 {
			s.dzU = g.DzCell[0]
		}
	}
	s.faces = make([]faceGeom, g.NZ+1)
	for f := 1; f < g.NZ; f++ {
		dUp := g.ZFace[f] - g.Z[f-1]
		dDown := g.Z[f] - g.ZFace[f]
		s.faces[f] = faceGeom{dUp: dUp, dDown: dDown, span: dUp + dDown,
			equal: s.uniform && dUp+dDown == s.dzU}
	}
	for i := range s.H {
		if math.IsNaN(s.H[i]) || math.IsInf(s.H[i], 0) {
			return nil, fmt.Errorf("initial head at layer %d is not finite", i)
		}
		s.Theta[i] = s.mat(i).WaterContent(s.H[i])
	}
	return s, nil
}

// NewSolverFromTheta builds a homogeneous solver from an initial
// water-content profile.
func NewSolverFromTheta(p constitutive.Params, g Grid, topKind string, pondedH float64,
	bottomKind string, initialTheta []float64, opts Options) (*Solver, error) {
	if err := checkThetaProfile([]constitutive.Params{p}, initialTheta, g.NZ, false); err != nil {
		return nil, err
	}
	heads := make([]float64, len(initialTheta))
	for i, th := range initialTheta {
		heads[i] = p.HeadFromWaterContent(th)
	}
	return NewSolver(p, g, topKind, pondedH, bottomKind, heads, opts)
}

// NewLayeredSolverFromTheta builds a layered solver from an initial
// water-content profile; each value is range-checked and inverted against
// that cell's own material bounds.
func NewLayeredSolverFromTheta(mats []constitutive.Params, g Grid, topKind string, pondedH float64,
	bottomKind string, initialTheta []float64, opts Options) (*Solver, error) {
	if err := checkThetaProfile(mats, initialTheta, g.NZ, true); err != nil {
		return nil, err
	}
	heads := make([]float64, len(initialTheta))
	for i, th := range initialTheta {
		heads[i] = mats[i].HeadFromWaterContent(th)
	}
	return NewLayeredSolver(mats, g, topKind, pondedH, bottomKind, heads, opts)
}

// checkThetaProfile validates per-cell initial water contents against each
// cell's own material bounds. When layered is false the single parameter set
// governs every cell.
func checkThetaProfile(mats []constitutive.Params, initialTheta []float64, nz int, layered bool) error {
	if len(initialTheta) != nz {
		return fmt.Errorf("initial profile length %d does not match grid size %d", len(initialTheta), nz)
	}
	for i, th := range initialTheta {
		var p constitutive.Params
		if layered {
			if len(mats) != nz {
				return fmt.Errorf("material profile length %d does not match grid size %d", len(mats), nz)
			}
			p = mats[i]
		} else {
			p = mats[0]
		}
		if math.IsNaN(th) || math.IsInf(th, 0) || th < p.ThetaR-1e-12 || th > p.ThetaS+1e-12 {
			return fmt.Errorf("%w: layer %d theta=%.6g outside [thetaR=%g, thetaS=%g]",
				ErrThetaOutOfRange, i, th, p.ThetaR, p.ThetaS)
		}
	}
	return nil
}

// mat returns the material governing cell i.
func (s *Solver) mat(i int) constitutive.Params {
	if s.Mats != nil {
		return s.Mats[i]
	}
	return s.Params
}

// cellDz returns the thickness of cell i.
func (s *Solver) cellDz(i int) float64 {
	if len(s.Grid.DzCell) > 0 {
		return s.Grid.DzCell[i]
	}
	return s.Grid.Dz
}

// MatAt and CellDz are read-only accessors used by the orchestration layer
// and tests to inspect the resolved per-cell constitutive structure.

// MatAt returns the material governing cell i.
func (s *Solver) MatAt(i int) constitutive.Params { return s.mat(i) }

// CellDz returns cell i's thickness [m].
func (s *Solver) CellDz(i int) float64 { return s.cellDz(i) }

// FaceFluxes evaluates all NZ+1 face fluxes (positive downward) for the
// given pressure-head profile, using the exact same face-flux expression as
// the Newton residual. A material interface carries one flux evaluated from
// both sides' own K(h) curves (series conductance), so the value seen from
// either adjacent cell is identical.
func (s *Solver) FaceFluxes(h []float64) []float64 {
	_, kCell := s.cellState(h)
	return s.fluxAt(h, kCell)
}

// State returns defensive copies of the current profile and cumulative
// accounting (used by the orchestration layer; never hands out internal
// slices so concurrent jobs cannot scribble on each other).
func (s *Solver) State() (h, theta []float64, time, cumTop, cumBot float64) {
	h = append([]float64(nil), s.H...)
	theta = append([]float64(nil), s.Theta...)
	return h, theta, s.Time, s.CumTop, s.CumBot
}

// Storage returns total water stored per unit area [m]: S = sum theta_i dz_i.
func (s *Solver) Storage() float64 {
	if s.uniform {
		sum := 0.0
		for _, th := range s.Theta {
			sum += th
		}
		return sum * s.dzU
	}
	sum := 0.0
	for i, th := range s.Theta {
		sum += th * s.cellDz(i)
	}
	return sum
}

// Steps returns the number of successfully completed time steps.
func (s *Solver) Steps() int { return s.steps }

// Failure is a structured compute failure (as opposed to an input
// validation error; see the job package for the wire representation).
type Failure struct {
	Kind    string
	Message string
}

// Failure kinds.
const (
	FailNonConvergence  = "NON_CONVERGENCE"
	FailThetaOutOfRange = "THETA_OUT_OF_RANGE"
	FailNonFinite       = "NON_FINITE_STATE"
)

func (f *Failure) Error() string { return f.Kind + ": " + f.Message }

// ErrThetaOutOfRange lets callers / tests match the bounds failure.
var ErrThetaOutOfRange = &Failure{Kind: FailThetaOutOfRange}

// StepResult reports one accepted time step.
type StepResult struct {
	TimeBefore    float64   `json:"time_before"` // [s]
	TimeAfter     float64   `json:"time_after"`  // [s]
	Dt            float64   `json:"dt"`          // [s]
	ThetaBefore   []float64 `json:"theta_before"`
	ThetaAfter    []float64 `json:"theta_after"`
	HBefore       []float64 `json:"h_before"`        // pressure heads [m]
	HAfter        []float64 `json:"h_after"`         // pressure heads [m]
	TopFlux       float64   `json:"top_flux"`        // step-end face flux [m/s], into column +
	BottomFlux    float64   `json:"bottom_flux"`     // [m/s], out of column +
	CumTopFlux    float64   `json:"cum_top_flux"`    // cumulative depth [m]
	CumBottomFlux float64   `json:"cum_bottom_flux"` // cumulative depth [m]
	StorageBefore float64   `json:"storage_before"`  // [m]
	StorageAfter  float64   `json:"storage_after"`   // [m]
	// MassBalanceResidual is StorageChange - (qTop - qBot)*dt [m]; it must be
	// zero within numerical precision on every accepted step.
	MassBalanceResidual float64 `json:"mass_balance_residual"`
	// MassBalanceRelative is the residual divided by the storage change
	// magnitude (0 when the change is ~0).
	MassBalanceRelative float64 `json:"mass_balance_relative"`
	Iterations          int     `json:"iterations"`
}

// thToSe converts a water content to effective saturation under p.
func thToSe(th float64, p constitutive.Params) float64 {
	if p.ThetaS <= p.ThetaR {
		return 0
	}
	se := (th - p.ThetaR) / (p.ThetaS - p.ThetaR)
	if se < 0 {
		return 0
	}
	if se > 1 {
		return 1
	}
	return se
}

// cellState evaluates each cell's water content and conductivity under its
// own material curves.
func (s *Solver) cellState(h []float64) (th, kCell []float64) {
	nz := s.Grid.NZ
	th = make([]float64, nz)
	kCell = make([]float64, nz)
	for i := 0; i < nz; i++ {
		p := s.mat(i)
		th[i] = p.WaterContent(h[i])
		kCell[i] = p.ConductivitySe(thToSe(th[i], p))
	}
	return th, kCell
}

// faceCoeffs evaluates one interior face's conductance coefficients.
//
// Darcy flux between two cell centres (z positive downward, total head
// H = h - z):
//
//	q = -Kf (H_d - H_u)/span = Kf + G*(h_u - h_d)
//
// with the two-point (series / harmonic) interface conductance
//
//	G = 1/(dUp/Ku + dDown/Kd),   Kf = span*G,
//
// where Ku = K_up(h_up) under the up-cell material curve and Kd likewise
// down-cell. There is one single q for the face, so continuity of the
// interface flux is built in even when the two materials' curves differ by
// orders of magnitude; the two heads stay independent. On a uniform
// equal-spaced face G = 2*Ku*Kd/(Ku+Kd)/dz — the classical harmonic mean —
// and the arithmetic-mean test switch applies.
func (s *Solver) faceCoeffs(f int, ku, kd float64) (kf, gamma float64) {
	fg := s.faces[f]
	if fg.equal {
		kf = s.mean.mean(ku, kd)
		return kf, kf / s.dzU
	}
	if ku <= 0 || kd <= 0 {
		return 0, 0
	}
	gamma = 1.0 / (fg.dUp/ku + fg.dDown/kd)
	return fg.span * gamma, gamma
}

// fluxAt evaluates all face fluxes qf[0..NZ] (positive downward) for the
// current iterate h and cell conductivities kCell.
func (s *Solver) fluxAt(h []float64, kCell []float64) []float64 {
	nz := s.Grid.NZ
	qf := make([]float64, nz+1)
	if s.uniform {
		dz := s.dzU
		for f := 1; f < nz; f++ {
			kf := s.mean.mean(kCell[f-1], kCell[f])
			// q = -Kf dH/dz ; H = h - z ; equally spaced centres dz apart.
			qf[f] = kf + (kf/dz)*(h[f-1]-h[f])
		}
		if s.TopKind == TopPondedHead {
			// boundary total head H_b = pondedH - 0; cell centre at z=dz/2.
			kHarm := s.mean.mean(s.Params.Ks, kCell[0])
			qf[0] = (2.0 * kHarm / dz) * (s.PondedH - h[0] + dz/2.0)
		}
		if s.BottomKind == BottomFreeDrainage {
			qf[nz] = kCell[nz-1] // unit-gradient drainage
		}
		return qf
	}

	for f := 1; f < nz; f++ {
		kf, g := s.faceCoeffs(f, kCell[f-1], kCell[f])
		qf[f] = kf + g*(h[f-1]-h[f])
	}
	// top face: series conductance between the ponded surface (K = Ks of the
	// surface material, centre distance dz0/2) and cell 0.
	if s.TopKind == TopPondedHead {
		dz0 := s.cellDz(0)
		kSurf := s.mat(0).Ks
		if kCell[0] <= 0 || kSurf <= 0 {
			qf[0] = 0
		} else {
			// centre-to-boundary distance dz0/2 is split into two equal
			// resistances; on a uniform cell this equals 2*harm(Ks,K0)/dz0.
			quarter := dz0 / 4.0
			g0 := 1.0 / (quarter/kSurf + quarter/kCell[0])
			qf[0] = g0 * (s.PondedH - h[0] + dz0/2.0)
		}
	}
	if s.BottomKind == BottomFreeDrainage {
		qf[nz] = kCell[nz-1] // unit-gradient drainage
	}
	return qf
}

// residual evaluates the finite-volume residual R_i [1/s] for iterate hIt,
// and also returns theta, cell K and face fluxes used to assemble the linear
// system.
//
//	R_i = (theta(hIt_i) - thetaOld_i)/dt - (qf_i - qf_{i+1})/dz_i
func (s *Solver) residual(hIt, thetaOld []float64, dt float64,
) (r []float64, th, kCell, qf []float64) {
	nz := s.Grid.NZ
	th, kCell = s.cellState(hIt)
	qf = s.fluxAt(hIt, kCell)
	r = make([]float64, nz)
	if s.uniform {
		dz := s.dzU
		for i := 0; i < nz; i++ {
			r[i] = (th[i]-thetaOld[i])/dt - (qf[i]-qf[i+1])/dz
		}
		return r, th, kCell, qf
	}
	for i := 0; i < nz; i++ {
		r[i] = (th[i]-thetaOld[i])/dt - (qf[i]-qf[i+1])/s.cellDz(i)
	}
	return r, th, kCell, qf
}

// Step advances one time step of length dt [s] with the fully implicit
// scheme, iterating to convergence. On failure the solver state is left
// untouched (the step is rejected outright; out-of-range values are never
// clipped into an answer).
func (s *Solver) Step(dt float64) (*StepResult, error) {
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, &Failure{Kind: FailNonConvergence, Message: fmt.Sprintf("invalid dt %g", dt)}
	}
	nz := s.Grid.NZ

	hOld := append([]float64(nil), s.H...)
	thetaOld := append([]float64(nil), s.Theta...)
	storageBefore := s.Storage()
	timeBefore := s.Time

	hIt := append([]float64(nil), hOld...)

	// Newton iteration with kink-aware line search. tryRelax starts from
	// the requested relaxation each iterate and is halved if the update
	// fails to reduce the residual.
	var (
		r, th, kCell, qf []float64
		iter             int
		maxDh            = math.Inf(1)
	)
	converged := false
	maxIter := s.Opts.MaxIterations
	for iter = 0; iter < maxIter; iter++ {
		r, th, kCell, qf = s.residual(hIt, thetaOld, dt)

		maxR := maxAbs(r)
		if hasNonFinite(r) {
			return nil, &Failure{Kind: FailNonFinite,
				Message: fmt.Sprintf("non-finite residual at iteration %d", iter)}
		}
		// Convergence needs both a small equation residual and a small head
		// increment (the first iterate skips the increment test).
		if maxR < s.Opts.ResidualTol && (iter == 0 || maxDh < s.Opts.HeadTol) {
			converged = true
			break
		}
		// Stagnation guard: residual essentially zero in absolute terms and
		// the update is tiny — accept (this happens at equilibria where one
		// cell sits right at the VG capacity cusp and chatter at round-off).
		if iter > 2 && maxR < stagnationResidual && maxDh < stagnationHead {
			converged = true
			break
		}

		// Assemble the full-Newton tridiagonal system J*dh = -G where
		// G_i = dz_i*R_i = dz_i*(th-thOld)/dt - qf_i + qf_{i+1}. Exact
		// derivatives of the (series/harmonic) face conductivities are
		// included, keeping quadratic convergence near a sharp wetting
		// front — including at material interfaces, where each side's
		// dK/dh enters through its own material curve.
		lower := make([]float64, nz) // J[i,i-1]
		diag := make([]float64, nz)  // J[i,i]
		upper := make([]float64, nz) // J[i,i+1]
		rhs := make([]float64, nz)

		// per-cell capacity and conductivity derivatives (own material)
		dkCell := make([]float64, nz)
		for i := 0; i < nz; i++ {
			dkCell[i] = s.mat(i).DKDh(hIt[i])
			if math.IsNaN(dkCell[i]) || math.IsInf(dkCell[i], 0) {
				dkCell[i] = 0
			}
		}
		// First put the storage term on the diagonal; zero-capacity
		// saturated cells get their diagonal from face conductance
		// couplings below.
		for i := 0; i < nz; i++ {
			capacity := s.mat(i).DThetaDH(hIt[i])
			if capacity < 0 || math.IsNaN(capacity) {
				capacity = 0
			}
			if s.uniform {
				diag[i] = s.dzU * capacity / dt
			} else {
				diag[i] = s.cellDz(i) * capacity / dt
			}
		}

		if s.uniform {
			dz := s.dzU
			// interior faces f = 1 .. nz-1:
			//   qf = Kf * [1 + (h_{f-1} - h_f)/dz]
			// R_i = dz*(th-thOld)/dt - qf_i + qf_{i+1}, so face f enters
			// row f-1 with +qf (its bottom face) and row f with -qf (its top).
			for f := 1; f < nz; f++ {
				ku, kd := kCell[f-1], kCell[f]
				kf := s.mean.mean(ku, kd)
				factor := 1.0 + (hIt[f-1]-hIt[f])/dz
				dqup := s.mean.dMeanDUp(ku, kd)*dkCell[f-1]*factor + kf/dz
				dqdn := s.mean.dMeanDDown(ku, kd)*dkCell[f]*factor - kf/dz
				diag[f-1] += dqup
				upper[f-1] += dqdn
				lower[f] += -dqup
				diag[f] += -dqdn
			}

			// top face
			if s.TopKind == TopPondedHead {
				// q0 = K0*(2/dz)*(hP - h0 + dz/2), K0 harmonic(Ks, K(h0));
				// -q0 enters row 0.
				k0Harm := s.mean.mean(s.Params.Ks, kCell[0])
				factor0 := s.PondedH - hIt[0] + dz/2.0
				dq0dn := s.mean.dMeanDDown(s.Params.Ks, kCell[0])*dkCell[0]*
					(2.0/dz)*factor0 - k0Harm*2.0/dz
				diag[0] += -dq0dn
			}
		} else {
			// General (possibly layered, non-uniform) interior faces.
			//	q = Kf + G*(hu-hd),  G = 1/(au/ad),  au = du/Ku + dd/Kd
			//	dG/dKu = G^2 * du/Ku^2 ; dG/dKd = G^2 * dd/Kd^2
			//	dq/dhu = dG/dKu*dKu/dhu*(hu-hd) + G
			//	dq/dhd = dG/dKd*dKd/dhd*(hu-hd) - G
			for f := 1; f < nz; f++ {
				fg := s.faces[f]
				ku, kd := kCell[f-1], kCell[f]
				var dqup, dqdn float64
				if ku > 0 && kd > 0 {
					au := fg.dUp/ku + fg.dDown/kd
					g := 1.0 / au
					dhdiff := hIt[f-1] - hIt[f]
					dgu := g * g * fg.dUp / (ku * ku)
					dgd := g * g * fg.dDown / (kd * kd)
					dqup = dgu*dkCell[f-1]*(fg.span+g*dhdiff) + g
					dqdn = dgd*dkCell[f]*(fg.span+g*dhdiff) - g
				}
				diag[f-1] += dqup
				upper[f-1] += dqdn
				lower[f] += -dqup
				diag[f] += -dqdn
			}

			// top face: G0 = 1/((dz0/4)/Ks + (dz0/4)/K0)
			// q0 = G0*(hP - h0 + dz0/2); -q0 enters row 0.
			if s.TopKind == TopPondedHead {
				dz0 := s.cellDz(0)
				kSurf := s.mat(0).Ks
				k0 := kCell[0]
				if k0 > 0 && kSurf > 0 {
					quarter := dz0 / 4.0
					a0 := quarter/kSurf + quarter/k0
					g0 := 1.0 / a0
					factor0 := s.PondedH - hIt[0] + dz0/2.0
					dg0dn := g0 * g0 * quarter / (k0 * k0)
					dq0dn := dg0dn*dkCell[0]*factor0 - g0
					diag[0] += -dq0dn
				}
			}
		}

		// bottom face: +qb enters row nz-1; free drainage qb = K(h_{nz-1}).
		if s.BottomKind == BottomFreeDrainage {
			diag[nz-1] += dkCell[nz-1]
		}

		if s.uniform {
			dz := s.dzU
			for i := 0; i < nz; i++ {
				rhs[i] = -dz * r[i]
			}
		} else {
			for i := 0; i < nz; i++ {
				rhs[i] = -s.cellDz(i) * r[i]
			}
		}

		// Regularise only rows that remain exactly uncoupled: saturated
		// cells with zero capacity at a closed boundary (e.g. an initially
		// hydrostatic profile with zero-flux top and bottom). A tiny
		// artificial compressibility makes the tridiagonal system solvable
		// without distorting actively coupled front cells; the actual state
		// and mass balance still use exact theta(h).
		for i := 0; i < nz; i++ {
			off := math.Abs(upper[i]) + math.Abs(lower[i])
			if off == 0 && diag[i] == 0 {
				if s.uniform {
					diag[i] = s.dzU * capacityFloor / dt
				} else {
					diag[i] = s.cellDz(i) * capacityFloor / dt
				}
			}
		}
		dh, ok := thomas(lower, diag, upper, rhs)
		if !ok {
			return nil, &Failure{Kind: FailNonFinite,
				Message: fmt.Sprintf("singular tridiagonal system at iteration %d", iter)}
		}
		maxDh = maxAbs(dh)

		// Newton trust bound: near a sharp front the Jacobian can be
		// ill-conditioned (saturated block meeting a nearly dry cell
		// through a tiny harmonic conductance) and produce physically
		// meaningless head corrections of tens of metres. Limiting the
		// single-iterate correction keeps the search in Newton's
		// convergence basin; the backtracking line search then takes over.
		tryRelax := s.Opts.Relaxation
		if maxDh > maxNewtonHeadStep {
			tryRelax = math.Min(tryRelax, maxNewtonHeadStep/maxDh)
		}

		// Backtracking (Armijo) line search on the smooth L2 residual
		// norm: the Newton step satisfies J dh = -G, so it is a descent
		// direction for ||R||; shrink until sufficient decrease is found.
		trial := make([]float64, nz)
		descent := false
		merit := l2Norm(r)
		var trialMerit float64
		for attempt := 0; attempt < 40; attempt++ {
			for i := range dh {
				trial[i] = hIt[i] + tryRelax*dh[i]
			}
			rr, _, _, _ := s.residual(trial, thetaOld, dt)
			trialMerit = l2Norm(rr)
			if !hasNonFinite(rr) && trialMerit < merit*(1.0-armijoC*tryRelax) {
				descent = true
				break
			}
			tryRelax *= 0.5
		}
		if hasNonFinite(trial) {
			return nil, &Failure{Kind: FailNonFinite,
				Message: fmt.Sprintf("non-finite trial state at iteration %d", iter)}
		}
		if !descent {
			// At a tight enough residual an infinitesimally small residual
			// reduction cannot be detected; accept via stagnation rules.
			if maxR >= stagnationResidual || maxDh >= stagnationHead {
				return nil, &Failure{Kind: FailNonConvergence, Message: fmt.Sprintf(
					"no descent direction at iteration %d: max|residual|=%g", iter, maxR)}
			}
			converged = true
			break
		}
		for i := range dh {
			hIt[i] = trial[i]
		}
	}

	// Recompute at the accepted iterate for bookkeeping.
	r, th, kCell, qf = s.residual(hIt, thetaOld, dt)
	maxR := maxAbs(r)
	if !converged && maxR >= s.Opts.ResidualTol {
		return nil, &Failure{Kind: FailNonConvergence, Message: fmt.Sprintf(
			"Newton iteration did not converge in %d iterations: max|residual|=%g",
			s.Opts.MaxIterations, maxR)}
	}
	if hasNonFinite(th) || hasNonFinite(hIt) {
		return nil, &Failure{Kind: FailNonFinite, Message: "non-finite accepted state"}
	}

	// Bounds guard: every accepted layer must stay inside the [thetaR,
	// thetaS] bounds of the material that governs it, and be finite.
	// Out-of-range is a hard failure, never clipped.
	for i, v := range th {
		if math.IsNaN(v) || math.IsInf(v, 0) ||
			math.IsNaN(hIt[i]) || math.IsInf(hIt[i], 0) {
			return nil, &Failure{Kind: FailNonFinite,
				Message: fmt.Sprintf("non-finite state at layer %d", i)}
		}
		p := s.mat(i)
		if v < p.ThetaR-1e-10 || v > p.ThetaS+1e-10 {
			return nil, &Failure{Kind: FailThetaOutOfRange, Message: fmt.Sprintf(
				"layer %d theta=%.10g outside [thetaR=%g, thetaS=%g] at t≈%g s",
				i, v, p.ThetaR, p.ThetaS, timeBefore+dt)}
		}
	}

	// Commit the accepted step.
	qTop := qf[0]
	qBot := qf[nz]
	s.H = hIt
	s.Theta = th
	s.Time = timeBefore + dt
	s.CumTop += qTop * dt
	s.CumBot += qBot * dt
	s.steps++

	storageAfter := s.Storage()
	storageChange := storageAfter - storageBefore
	netFluxDepth := (qTop - qBot) * dt
	imbalance := storageChange - netFluxDepth
	rel := 0.0
	if d := math.Abs(storageChange); d > 1e-14 {
		rel = imbalance / d
	}

	return &StepResult{
		TimeBefore:          timeBefore,
		TimeAfter:           s.Time,
		Dt:                  dt,
		ThetaBefore:         thetaOld,
		ThetaAfter:          append([]float64(nil), th...),
		HBefore:             hOld,
		HAfter:              append([]float64(nil), hIt...),
		TopFlux:             qTop,
		BottomFlux:          qBot,
		CumTopFlux:          s.CumTop,
		CumBottomFlux:       s.CumBot,
		StorageBefore:       storageBefore,
		StorageAfter:        storageAfter,
		MassBalanceResidual: imbalance,
		MassBalanceRelative: rel,
		Iterations:          iter,
	}, nil
}

// thomas solves a tridiagonal system A x = d.
// lower[i], diag[i], upper[i] hold the coefficients of row i (lower[0]
// and upper[n-1] are unused). Returns ok=false on a singular matrix.
func thomas(lower, diag, upper, d []float64) ([]float64, bool) {
	n := len(diag)
	cp := make([]float64, n)
	dp := make([]float64, n)
	const floor = 1e-30
	if math.Abs(diag[0]) < floor {
		return nil, false
	}
	cp[0] = upper[0] / diag[0]
	dp[0] = d[0] / diag[0]
	for i := 1; i < n; i++ {
		denom := diag[i] - lower[i]*cp[i-1]
		if math.Abs(denom) < floor || math.IsNaN(denom) || math.IsInf(denom, 0) {
			return nil, false
		}
		if i < n-1 {
			cp[i] = upper[i] / denom
		}
		dp[i] = (d[i] - lower[i]*dp[i-1]) / denom
	}
	x := make([]float64, n)
	x[n-1] = dp[n-1]
	for i := n - 2; i >= 0; i-- {
		x[i] = dp[i] - cp[i]*x[i+1]
	}
	return x, true
}

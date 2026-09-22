// Package solver advances the mixed (h-based) Richards equation on a
// one-dimensional vertical soil column:
//
//	dtheta/dt = d/dz [ K(h) * (dh/dz - 1) ],   z positive downward,
//
// discretised with a cell-centred finite-volume grid, fully implicit
// (backward Euler) time stepping and a full-Newton iteration (exact
// Jacobian of the face conductivities, backtracking line search) per time
// step. Inter-cell conductivities are harmonic means.
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

// Solver is the per-job stateful time marcher. A Solver instance belongs to
// exactly one infiltration job; nothing is shared between instances.
type Solver struct {
	Params     constitutive.Params
	Grid       Grid
	Opts       Options
	TopKind    string
	PondedH    float64 // surface pressure head [m] when TopKind == TopPondedHead
	BottomKind string

	// current state
	H     []float64 // pressure heads at cell centres [m]
	Theta []float64 // volumetric water contents
	Time  float64   // accumulated simulation time [s]

	// cumulative boundary water depths [m] (flux integrated over time),
	// sign: CumTop inflow positive, CumBot outflow positive.
	CumTop float64
	CumBot float64

	steps int

	// mean selects inter-block K averaging; harmonic in production.
	mean interblockMean
}

// NewSolver builds a solver from an initial pressure-head profile.
func NewSolver(p constitutive.Params, g Grid, topKind string, pondedH float64,
	bottomKind string, initialHeads []float64, opts Options) (*Solver, error) {
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
	s := &Solver{
		Params: p, Grid: g, Opts: opts,
		TopKind: topKind, PondedH: pondedH, BottomKind: bottomKind,
		H:     append([]float64(nil), initialHeads...),
		Theta: make([]float64, g.NZ),
		mean:  HarmonicMean,
	}
	for i := range s.H {
		if math.IsNaN(s.H[i]) || math.IsInf(s.H[i], 0) {
			return nil, fmt.Errorf("initial head at layer %d is not finite", i)
		}
		s.Theta[i] = p.WaterContent(s.H[i])
	}
	return s, nil
}

// NewSolverFromTheta builds a solver from an initial water-content profile.
func NewSolverFromTheta(p constitutive.Params, g Grid, topKind string, pondedH float64,
	bottomKind string, initialTheta []float64, opts Options) (*Solver, error) {
	if len(initialTheta) != g.NZ {
		return nil, fmt.Errorf("initial profile length %d does not match grid size %d", len(initialTheta), g.NZ)
	}
	for i, th := range initialTheta {
		if math.IsNaN(th) || math.IsInf(th, 0) || th < p.ThetaR-1e-12 || th > p.ThetaS+1e-12 {
			return nil, fmt.Errorf("%w: layer %d theta=%.6g outside [thetaR=%g, thetaS=%g]",
				ErrThetaOutOfRange, i, th, p.ThetaR, p.ThetaS)
		}
	}
	heads := make([]float64, len(initialTheta))
	for i, th := range initialTheta {
		heads[i] = p.HeadFromWaterContent(th)
	}
	return NewSolver(p, g, topKind, pondedH, bottomKind, heads, opts)
}

// State returns defensive copies of the current profile and cumulative
// accounting (used by the orchestration layer; never hands out internal
// slices so concurrent jobs cannot scribble on each other).
func (s *Solver) State() (h, theta []float64, time, cumTop, cumBot float64) {
	h = append([]float64(nil), s.H...)
	theta = append([]float64(nil), s.Theta...)
	return h, theta, s.Time, s.CumTop, s.CumBot
}

// Storage returns total water stored per unit area [m]: S = sum theta_i dz.
func (s *Solver) Storage() float64 {
	sum := 0.0
	for _, th := range s.Theta {
		sum += th
	}
	return sum * s.Grid.Dz
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
	HAfter        []float64 `json:"h_after"`         // [m]
	TopFlux       float64   `json:"top_flux"`        // step-end face flux [m/s], into column +
	BottomFlux    float64   `json:"bottom_flux"`     // [m/s], out of column +
	CumTopFlux    float64   `json:"cum_top_flux"`    // cumulative depth [m]
	CumBottomFlux float64   `json:"cum_bottom_flux"` // [m]
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

// faceConductances fills gamma[f] = Kf/dz-like conductance for each of the
// NZ+1 faces.
//
// gamma[f] for interior faces f=1..NZ-1: harmonic-mean K / dz.
// gamma[0]: 0 — the ponded-top face is state dependent (harmonic mean of
// surface Ks and cell-0 K over the half-cell), assembled in fluxAt.
// gamma[NZ]: 0 (bottom flux is evaluated explicitly in fluxAt).
func (s *Solver) faceConductances(kCell []float64) (gamma []float64) {
	nz := s.Grid.NZ
	dz := s.Grid.Dz
	gamma = make([]float64, nz+1)
	for f := 1; f < nz; f++ {
		kf := s.mean.mean(kCell[f-1], kCell[f])
		gamma[f] = kf / dz
	}
	if s.TopKind == TopPondedHead {
		// state-dependent surface conductance
		gamma[0] = s.topConductance(kCell[0])
	}
	return gamma
}

// topConductance returns the surface-to-cell-0 conductance for a ponded
// boundary: harmonic mean of the ponded-surface conductivity (Ks) and the
// top-cell conductivity, across the half-cell distance dz/2.
func (s *Solver) topConductance(k0 float64) float64 {
	kSurf := s.Params.Ks
	kHarm := s.mean.mean(kSurf, k0)
	return 2.0 * kHarm / s.Grid.Dz
}

// fluxAt evaluates all face fluxes qf[0..NZ] (positive downward) for the
// current iterate h and cell conductivities kCell.
func (s *Solver) fluxAt(h []float64, kCell []float64, gamma []float64) []float64 {
	nz := s.Grid.NZ
	dz := s.Grid.Dz
	qf := make([]float64, nz+1)
	for f := 1; f < nz; f++ {
		kf := s.mean.mean(kCell[f-1], kCell[f])
		// q = -Kf dH/dz ; H = h - z ; equally spaced centres dz apart.
		qf[f] = kf + gamma[f]*(h[f-1]-h[f])
	}
	// top face
	switch s.TopKind {
	case TopPondedHead:
		// boundary total head H_b = pondedH - 0; cell centre at z=dz/2.
		qf[0] = gamma[0] * (s.PondedH - h[0] + dz/2.0)
	case TopZeroFlux:
		qf[0] = 0
	}
	// bottom face
	switch s.BottomKind {
	case BottomFreeDrainage:
		qf[nz] = kCell[nz-1] // unit-gradient drainage
	case BottomZeroFlux:
		qf[nz] = 0
	}
	return qf
}

// residual evaluates the finite-volume residual R_i [1/s] for iterate hIt,
// and also returns theta, cell K, conductances and face fluxes used to
// assemble the linear system.
//
//	R_i = (theta(hIt_i) - thetaOld_i)/dt - (qf_i - qf_{i+1})/dz
func (s *Solver) residual(hIt, thetaOld []float64, dt float64,
) (r []float64, th, kCell, gamma, qf []float64) {
	nz := s.Grid.NZ
	dz := s.Grid.Dz
	th = make([]float64, nz)
	kCell = make([]float64, nz)
	for i := 0; i < nz; i++ {
		th[i] = s.Params.WaterContent(hIt[i])
		kCell[i] = s.Params.ConductivitySe(thToSe(th[i], s.Params))
	}
	gamma = s.faceConductances(kCell)
	qf = s.fluxAt(hIt, kCell, gamma)
	r = make([]float64, nz)
	for i := 0; i < nz; i++ {
		r[i] = (th[i]-thetaOld[i])/dt - (qf[i]-qf[i+1])/dz
	}
	return r, th, kCell, gamma, qf
}

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

// Step advances one time step of length dt [s] with the fully implicit
// scheme, iterating to convergence. On failure the solver state is left
// untouched (the step is rejected outright; out-of-range values are never
// clipped into an answer).
func (s *Solver) Step(dt float64) (*StepResult, error) {
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, &Failure{Kind: FailNonConvergence, Message: fmt.Sprintf("invalid dt %g", dt)}
	}
	nz := s.Grid.NZ
	dz := s.Grid.Dz

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
		r, th, kCell, _, qf = s.residual(hIt, thetaOld, dt)

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
		// G_i = dz*R_i = dz*(th-thOld)/dt - qf_i + qf_{i+1}. Exact
		// derivatives of the (harmonic-mean) face conductivities are
		// included, keeping quadratic convergence near a sharp wetting
		// front where Picard (frozen K) only converges linearly.
		lower := make([]float64, nz) // J[i,i-1]
		diag := make([]float64, nz)  // J[i,i]
		upper := make([]float64, nz) // J[i,i+1]
		rhs := make([]float64, nz)

		// per-cell capacity and conductivity derivatives
		dkCell := make([]float64, nz)
		for i := 0; i < nz; i++ {
			dkCell[i] = s.Params.DKDh(hIt[i])
			if math.IsNaN(dkCell[i]) || math.IsInf(dkCell[i], 0) {
				dkCell[i] = 0
			}
		}
		// First put the storage term on the diagonal; zero-capacity
		// saturated cells get their diagonal from face conductance
		// couplings below.
		for i := 0; i < nz; i++ {
			capacity := s.Params.DThetaDH(hIt[i])
			if capacity < 0 || math.IsNaN(capacity) {
				capacity = 0
			}
			diag[i] = dz * capacity / dt
		}

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
		switch s.TopKind {
		case TopPondedHead:
			// q0 = K0*(2/dz)*(hP - h0 + dz/2), K0 harmonic(Ks, K(h0));
			// -q0 enters row 0.
			k0Harm := s.mean.mean(s.Params.Ks, kCell[0])
			factor0 := s.PondedH - hIt[0] + dz/2.0
			dq0dn := s.mean.dMeanDDown(s.Params.Ks, kCell[0])*dkCell[0]*
				(2.0/dz)*factor0 - k0Harm*2.0/dz
			diag[0] += -dq0dn
		case TopZeroFlux:
			// q0 = 0
		}

		// bottom face: +qb enters row nz-1; free drainage qb = K(h_{nz-1}).
		if s.BottomKind == BottomFreeDrainage {
			diag[nz-1] += dkCell[nz-1]
		}

		for i := 0; i < nz; i++ {
			rhs[i] = -dz * r[i]
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
				diag[i] = dz * capacityFloor / dt
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
			rr, _, _, _, _ := s.residual(trial, thetaOld, dt)
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
	r, th, _, _, qf = s.residual(hIt, thetaOld, dt)
	maxR := maxAbs(r)
	if !converged && maxR >= s.Opts.ResidualTol {
		return nil, &Failure{Kind: FailNonConvergence, Message: fmt.Sprintf(
			"Picard iteration did not converge in %d iterations: max|residual|=%g",
			s.Opts.MaxIterations, maxR)}
	}
	if hasNonFinite(th) || hasNonFinite(hIt) {
		return nil, &Failure{Kind: FailNonFinite, Message: "non-finite accepted state"}
	}

	// Bounds guard: every accepted layer must stay inside [thetaR, thetaS]
	// and be finite. Out-of-range is a hard failure, never clipped.
	for i, v := range th {
		if math.IsNaN(v) || math.IsInf(v, 0) ||
			math.IsNaN(hIt[i]) || math.IsInf(hIt[i], 0) {
			return nil, &Failure{Kind: FailNonFinite,
				Message: fmt.Sprintf("non-finite state at layer %d", i)}
		}
		if v < s.Params.ThetaR-1e-10 || v > s.Params.ThetaS+1e-10 {
			return nil, &Failure{Kind: FailThetaOutOfRange, Message: fmt.Sprintf(
				"layer %d theta=%.10g outside [thetaR=%g, thetaS=%g] at t≈%g s",
				i, v, s.Params.ThetaR, s.Params.ThetaS, timeBefore+dt)}
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

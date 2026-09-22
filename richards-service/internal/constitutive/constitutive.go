// Package constitutive implements the van Genuchten water-retention model
// together with the Mualem relative-permeability model used by the Richards
// solver.
//
// Coordinate/sign convention used throughout the service:
//   - depth z is positive downward;
//   - pressure head h is negative in the unsaturated zone, h >= 0 saturated;
//   - everything in SI units (m, s, K in m/s).
package constitutive

import (
	"math"
)

// PoresConnectivityL is the fixed Mualem tortuosity/connectivity exponent
// (the "reduction factor" l = 1/2).
const PoresConnectivityL = 0.5

// SaturationSmoothingHead is the width of the narrow head band just below
// h = 0 over which the VG retention curve is made C1-continuous with its
// flat saturated branch. At h = 0, Se is still exactly 1; only the capacity
// is smoothed from 0 (flat branch) to the VG branch value over 2 mm, which
// removes the slope kink that otherwise stalls a Newton iteration whenever
// a wetting cell crosses saturation.
const SaturationSmoothingHead = 2e-3

// Params holds the van Genuchten–Mualem constitutive parameters.
//
//	Se(h)            = [1 + (alpha*|h|)^n]^(-m),   m = 1 - 1/n      (h < 0)
//	Se(h)            = 1                                             (h >= 0)
//	theta            = thetaR + (thetaS - thetaR) * Se
//	Kr(Se)           = Se^l * [1 - (1 - Se^(1/m))^m]^2
//	K(Se)            = Ks * Kr
type Params struct {
	// Alpha is the inverse air-entry (bubbling) pressure head [1/m], alpha > 0.
	Alpha float64 `json:"alpha"`
	// N is the pore-size-distribution index, n > 1; m is locked to 1-1/n.
	N float64 `json:"n"`
	// ThetaR is residual volumetric water content, 0 <= thetaR < thetaS.
	ThetaR float64 `json:"theta_r"`
	// ThetaS is saturated volumetric water content, thetaS > thetaR.
	ThetaS float64 `json:"theta_s"`
	// Ks is saturated hydraulic conductivity [m/s], Ks > 0.
	Ks float64 `json:"ks"`
}

// M returns the locked exponent m = 1 - 1/n.
func (p Params) M() float64 { return 1.0 - 1.0/p.N }

// vgSe is the bare van Genuchten branch (no smoothing).
func (p Params) vgSe(h float64) float64 {
	x := p.Alpha * (-h)
	return math.Pow(1.0+math.Pow(x, p.N), -p.M())
}

// vgDSe is dSe/dh of the bare VG branch.
func (p Params) vgDSe(h float64) float64 {
	x := p.Alpha * (-h)
	base := 1.0 + math.Pow(x, p.N)
	return p.Alpha * p.M() * p.N * math.Pow(x, p.N-1.0) * math.Pow(base, -(p.M()+1.0))
}

// EffectiveSaturation returns Se as a function of pressure head h (m).
// For h >= 0, Se = 1; for -delta < h < 0 the bare VG curve is C1-blended
// into the flat saturated branch (see SaturationSmoothingHead) with a cubic
// Hermite weight: at both ends of the band both value and slope match, so
// the smoothed curve is globally C1 while Se(0) = 1 exactly.
func (p Params) EffectiveSaturation(h float64) float64 {
	if h >= 0 {
		return 1
	}
	seRaw := p.vgSe(h)
	if h >= -SaturationSmoothingHead {
		s := -h / SaturationSmoothingHead // s = 1 at -delta, 0 at 0
		w := hermiteWeight(s)             // 0 at s=0, 1 at s=1, w'=0 at both
		return 1.0 - w*(1.0-seRaw)
	}
	return seRaw
}

// dSeDHead is dSe/dh including the C1 smoothing band.
func (p Params) dSeDHead(h float64) float64 {
	if h >= 0 {
		return 0
	}
	seRaw := p.vgSe(h)
	if h >= -SaturationSmoothingHead {
		s := -h / SaturationSmoothingHead
		w := hermiteWeight(s)
		dwDh := hermiteWeightDerivS(s) * (-1.0 / SaturationSmoothingHead)
		dRaw := p.vgDSe(h)
		// Se = 1 - w(1-SeRaw)
		// dSe/dh = -[w'(1-SeRaw) - w*dSeRaw/dh]
		return -dwDh*(1.0-seRaw) + w*dRaw
	}
	return p.vgDSe(h)
}

// hermiteWeight: w(0)=0, w(1)=1, w'(0)=w'(1)=0; smoothstep cubic.
func hermiteWeight(s float64) float64 {
	if s <= 0 {
		return 0
	}
	if s >= 1 {
		return 1
	}
	return s * s * (3 - 2*s)
}

// dw/ds of the cubic Hermite weight.
func hermiteWeightDerivS(s float64) float64 {
	if s <= 0 || s >= 1 {
		return 0
	}
	return 6 * s * (1 - s)
}

// WaterContent returns theta = thetaR + (thetaS-thetaR)*Se.
// The returned value always lies in [thetaR, thetaS] for finite input.
func (p Params) WaterContent(h float64) float64 {
	return p.ThetaR + (p.ThetaS-p.ThetaR)*p.EffectiveSaturation(h)
}

// Conductivity returns the unsaturated hydraulic conductivity K(h) [m/s].
func (p Params) Conductivity(h float64) float64 {
	se := p.EffectiveSaturation(h)
	return p.Ks * RelativePermeability(se, p.M())
}

// DKDh returns dK/dh [m/s per m] for the Newton Jacobian. K is constant
// (Ks) for h >= 0; for h < 0 the chain rule dK/dh = Ks dKr/dSe * dSe/dh is
// used.
func (p Params) DKDh(h float64) float64 {
	if h >= 0 {
		return 0
	}
	se := p.EffectiveSaturation(h)
	return p.Ks * dKrDSe(se, p.M()) * p.dSeDHead(h)
}

// dKrDSe analytically differentiates the Mualem model
// Kr = Se^l [1 - (1-Se^(1/m))^m]^2.
func dKrDSe(se, m float64) float64 {
	se = clamp01(se)
	if se <= 0 || se >= 1 {
		return 0
	}
	a := math.Pow(se, 1.0/m) // a = Se^(1/m)
	var b float64            // b = (1-a)^m
	if a < 0.5 {
		b = math.Exp(m * math.Log1p(-a))
	} else {
		b = math.Pow(1.0-a, m)
	}
	// da/dSe = Se^(1/m - 1)/m = a/(m*Se)
	// db/dSe = -m(1-a)^(m-1) * da/dSe
	daDSe := a / (m * se)
	dbDSe := -m * math.Pow(1.0-a, m-1.0) * daDSe
	// Kr = Se^l (1-b)^2
	g := 1.0 - b
	kr := math.Pow(se, PoresConnectivityL) * g * g
	dKr := PoresConnectivityL*kr/se + math.Pow(se, PoresConnectivityL)*2.0*g*(-dbDSe)
	if math.IsNaN(dKr) || math.IsInf(dKr, 0) {
		return 0
	}
	return dKr
}

// ConductivitySe returns K for a given effective saturation.
func (p Params) ConductivitySe(se float64) float64 {
	return p.Ks * RelativePermeability(clamp01(se), p.M())
}

// RelativePermeability evaluates the Mualem model
// Kr = Se^l * [1 - (1 - Se^(1/m))^m]^2 in a numerically stable way.
//
// For very small Se the bracket is evaluated as
// (1-Se^(1/m))^m = exp(m*log1p(-Se^(1/m))), and the limiting Kr -> 0.
func RelativePermeability(se, m float64) float64 {
	se = clamp01(se)
	if se <= 0 {
		return 0
	}
	if se >= 1 {
		return 1
	}
	a := math.Pow(se, 1.0/m) // a = Se^(1/m), in (0,1)
	var bracket float64
	if a < 0.5 {
		bracket = math.Exp(m * math.Log1p(-a))
	} else {
		bracket = math.Pow(1.0-a, m)
	}
	kr := math.Pow(se, PoresConnectivityL) * (1.0 - bracket) * (1.0 - bracket)
	if math.IsNaN(kr) || math.IsInf(kr, 0) || kr < 0 || kr > 1.0+1e-12 {
		// Should never happen; guard rather than emit garbage.
		if kr > 1 {
			return 1
		}
		return 0
	}
	return kr
}

// DThetaDH returns the specific water-capacity C(h) = d theta/d h.
//
// Away from saturation it is the analytic van Genuchten capacity; within
// the narrow smoothing band (see SaturationSmoothingHead) the C1-blended
// derivative is used, and for h >= 0 the flat saturated branch gives
// C = 0. The solver separately regularises the resulting zero storage where
// a closed boundary makes the system singular.
func (p Params) DThetaDH(h float64) float64 {
	if h >= 0 {
		return 0
	}
	return (p.ThetaS - p.ThetaR) * p.dSeDHead(h)
}

// HeadFromWaterContent inverts the retention curve: given theta in
// [thetaR, thetaS] return the pressure head h.
//   - theta >= thetaS            -> h = 0
//   - theta <= thetaR            -> h = -1/alpha * HugeAlphaX
//     (representative "very dry" head; thetaR is an asymptote)
func (p Params) HeadFromWaterContent(theta float64) float64 {
	m := p.M()
	if theta >= p.ThetaS {
		return 0
	}
	if theta <= p.ThetaR {
		// Se -> 0 asymptote; pick a finite representative dry head.
		return -DryHeadAlphaX / p.Alpha
	}
	se := (theta - p.ThetaR) / (p.ThetaS - p.ThetaR)
	se = clamp01(se)
	// Se = (1 + (alpha*|h|)^n)^(-m)
	// Se^(-1/m) = 1 + (alpha*|h|)^n
	xpow := math.Pow(se, -1.0/m) - 1.0
	if xpow <= 0 {
		return 0
	}
	x := math.Pow(xpow, 1.0/p.N) // alpha*|h|
	return -(x / p.Alpha)
}

// DryHeadAlphaX is the value of alpha*|h| used to represent the residual
// (asymptotic) water content. Large enough that Se is essentially zero.
const DryHeadAlphaX = 1e6

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

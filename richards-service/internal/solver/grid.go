package solver

import (
	"fmt"
	"math"
)

// Grid is the cell-centred finite-volume discretisation of the vertical
// soil column. Depth z is positive downward from the soil surface.
//
// A grid built with NewGrid is uniform: every cell has thickness Dz. A grid
// built with NewLayeredGrid is piecewise-uniform along soil-material
// segments: cells inside one segment are equally thick, but thickness may
// differ between segments, and every material interface coincides exactly
// with a cell face (no cell straddles two materials). Uniform is true only
// for a genuinely equidistant grid; the uniform code paths in the solver
// are then bit-identical to the legacy single-material implementation.
type Grid struct {
	NZ     int       // number of cells (layers)
	Dz     float64   // uniform cell thickness [m]; 0 when !Uniform
	Depth  float64   // total column thickness [m]
	Uniform bool     // every cell has the same thickness
	Z      []float64 // nodal (cell-centre) depths, length NZ, [m]
	ZFace  []float64 // interface depths, length NZ+1; ZFace[0]=0, ZFace[NZ]=Depth
	Dzs    []float64 // per-cell thickness, length NZ [m]
	// MaterialOf[i] is the material-segment index owning cell i; nil for a
	// single-material grid (equivalent to all zeros).
	MaterialOf []int
}

// NewGrid builds a uniform cell-centred grid of nz cells over a column of
// total thickness depth [m].
func NewGrid(nz int, depth float64) Grid {
	dz := depth / float64(nz)
	g := Grid{NZ: nz, Dz: dz, Depth: depth, Uniform: true,
		Z: make([]float64, nz), ZFace: make([]float64, nz+1),
		Dzs: make([]float64, nz)}
	for i := 0; i <= nz; i++ {
		g.ZFace[i] = float64(i) * dz
	}
	for i := 0; i < nz; i++ {
		g.Z[i] = (float64(i) + 0.5) * dz
		g.Dzs[i] = dz
	}
	return g
}

// NewLayeredGrid builds a piecewise-uniform cell-centred grid whose faces
// are aligned with the material interfaces.
//
// segThickness[k] is the thickness [m] of material segment k (top to
// bottom), segCells[k] the number of equally thick cells allocated inside
// it. Every segment must have positive thickness and at least one cell;
// the returned grid's faces hit each interface exactly. When every segment
// happens to share the same cell thickness, the uniform grid is rebuilt
// through NewGrid so that uniform/layered constructions stay numerically
// identical.
func NewLayeredGrid(segThickness []float64, segCells []int) (Grid, error) {
	ns := len(segThickness)
	if ns < 1 {
		return Grid{}, fmt.Errorf("soil profile needs at least one material segment")
	}
	if len(segCells) != ns {
		return Grid{}, fmt.Errorf("got %d segment thicknesses but %d cell counts",
			ns, len(segCells))
	}
	nz := 0
	depth := 0.0
	allDz := math.Inf(1)
	uniformizable := true
	for k := 0; k < ns; k++ {
		if math.IsNaN(segThickness[k]) || math.IsInf(segThickness[k], 0) ||
			segThickness[k] <= 0 {
			return Grid{}, fmt.Errorf("segment %d thickness must be > 0, got %g",
				k, segThickness[k])
		}
		if segCells[k] < 1 {
			return Grid{}, fmt.Errorf("segment %d must contain at least one cell, got %d",
				k, segCells[k])
		}
		nz += segCells[k]
		depth += segThickness[k]
		dzk := segThickness[k] / float64(segCells[k])
		if k == 0 {
			allDz = dzk
		} else if dzk != allDz {
			uniformizable = false
		}
	}
	if uniformizable {
		// Identical cell thickness everywhere: rebuild the single-equidistant
		// grid so face depths carry no accumulated per-segment round-off.
		g := NewGrid(nz, depth)
		g.MaterialOf = materialOf(segCells, nz)
		return g, nil
	}

	g := Grid{NZ: nz, Dz: 0, Depth: depth, Uniform: false,
		Z: make([]float64, nz), ZFace: make([]float64, nz+1),
		Dzs: make([]float64, nz), MaterialOf: materialOf(segCells, nz)}
	// Faces are constructed per segment so that the last face of a segment
	// lands exactly on the (cumulative) interface depth; the first face of
	// the next segment then starts from that same depth.
	i := 0
	face := 0.0
	g.ZFace[0] = 0
	interfaceDepth := 0.0
	for k := 0; k < ns; k++ {
		dzk := segThickness[k] / float64(segCells[k])
		for j := 0; j < segCells[k]; j++ {
			g.Dzs[i] = dzk
			g.Z[i] = face + 0.5*dzk
			face += dzk
			i++
			g.ZFace[i] = face
		}
		interfaceDepth += segThickness[k]
		// Pin the boundary face exactly onto the interface depth.
		g.ZFace[i] = interfaceDepth
		face = interfaceDepth
	}
	return g, nil
}

func materialOf(segCells []int, nz int) []int {
	of := make([]int, nz)
	i := 0
	for k, n := range segCells {
		for j := 0; j < n; j++ {
			of[i] = k
			i++
		}
	}
	return of
}

// MaterialIndex returns the material-segment index owning cell i: 0 for a
// single-material grid, MaterialOf[i] for a layered one.
func (g Grid) MaterialIndex(i int) int {
	if g.MaterialOf == nil {
		return 0
	}
	return g.MaterialOf[i]
}

// faceGeometry returns, for interior face f (1..NZ-1), the distances from
// the centres of the up and down cells to the face, and the centre-to-
// centre distance over which the inter-cell head gradient is evaluated.
func (g Grid) faceGeometry(f int) (lUp, lDown, lFace float64) {
	lUp = 0.5 * g.Dzs[f-1]
	lDown = 0.5 * g.Dzs[f]
	return lUp, lDown, lUp + lDown
}

// interblockMean selects the inter-cell hydraulic conductivity.
//
// Production computations always use the harmonic mean, as required:
// conductivity spans orders of magnitude near a wetting front and the
// arithmetic mean overestimates the inter-block flux, distorting front
// speed and direction. ArithmeticMean exists solely so the test suite can
// demonstrate the two are not interchangeable.
//
// The mean is used for uniform grids (and, equivalently, across any
// interface whose adjacent cell centres are equally far from the face);
// non-uniform material interfaces use the distance-weighted harmonic mean
// weightedFaceK below. Both expressions coincide exactly when the two
// half-distances are equal.
type interblockMean int

const (
	HarmonicMean interblockMean = iota
	ArithmeticMean
)

func (m interblockMean) mean(kUp, kDown float64) float64 {
	switch m {
	case ArithmeticMean:
		return 0.5 * (kUp + kDown)
	default:
		// harmonic mean of two equal-spaced adjacent cells:
		// 2*k1*k2/(k1+k2); degrade gracefully at the zero limit.
		s := kUp + kDown
		if s <= 0 {
			return 0
		}
		return 2.0 * kUp * kDown / s
	}
}

// dMeanDUp / dMeanDDown return derivatives of the interblock mean with
// respect to the up/down cell conductivity (used by the Newton Jacobian).
func (m interblockMean) dMeanDUp(kUp, kDown float64) float64 {
	switch m {
	case ArithmeticMean:
		return 0.5
	default:
		s := kUp + kDown
		if s <= 0 {
			return 0
		}
		return 2.0 * kDown * kDown / (s * s)
	}
}

func (m interblockMean) dMeanDDown(kUp, kDown float64) float64 {
	switch m {
	case ArithmeticMean:
		return 0.5
	default:
		s := kUp + kDown
		if s <= 0 {
			return 0
		}
		return 2.0 * kUp * kUp / (s * s)
	}
}

// weightedFaceK is the distance-weighted harmonic-mean interface
// conductivity between two adjacent cells whose centres are lUp and lDown
// away from their common face:
//
//	Kf = (lUp + lDown) / (lUp/KUp + lDown/KDown)
//	   = (lUp+lDown)*KUp*KDown / (lDown*KUp + lUp*KDown)
//
// This is the correct inter-block conductivity on a non-uniform grid and,
// crucially, at a material interface: each side contributes exactly its
// own half-cell resistance (computed from its own K(h) curve), so the
// single shared face value is the flux-continuous interface conductivity.
// Pressure head and water content are free to jump across the face; only
// the flux is forced to be continuous.
//
// With lUp == lDown this reduces to 2*KUp*KDown/(KUp+KDown); the uniform
// solver path keeps the unweighted expression literally to preserve
// bit-identical single-material results.
func weightedFaceK(kUp, kDown, lUp, lDown float64) float64 {
	den := lDown*kUp + lUp*kDown
	if den <= 0 {
		return 0
	}
	return (lUp + lDown) * kUp * kDown / den
}

// dWeightedFaceKDUp / dWeightedFaceKDDown analytically differentiate
// weightedFaceK with respect to either cell conductivity (Newton Jacobian
// across a non-uniform / material-interface face).
//
//	Kf = N/D, N = L*ku*kd, D = ld*ku + lu*kd, L = lu+ld
//	dKf/dku = L*kd*(D - ku*ld)/D^2 = L*lu*kd^2/D^2
//	dKf/dkd = L*ld*ku^2/D^2
func dWeightedFaceKDUp(kUp, kDown, lUp, lDown float64) float64 {
	den := lDown*kUp + lUp*kDown
	if den <= 0 {
		return 0
	}
	return (lUp + lDown) * lUp * kDown * kDown / (den * den)
}

func dWeightedFaceKDDown(kUp, kDown, lUp, lDown float64) float64 {
	den := lDown*kUp + lUp*kDown
	if den <= 0 {
		return 0
	}
	return (lUp + lDown) * lDown * kUp * kUp / (den * den)
}

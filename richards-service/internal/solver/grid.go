package solver

import "fmt"

// Grid is the cell-centred finite-volume discretisation of the vertical
// soil column. Nodal depth z_i is the centre of cell i, counted positive
// downward from the soil surface (z = 0).
//
// A grid is either uniform (DzCell == nil; every cell is Dz thick) or
// piecewise-uniform (DzCell holds one thickness per cell), which is what a
// layered material profile produces: each material segment is subdivided
// into equal-thickness cells, and segment interfaces coincide exactly with
// a face in ZFace — no cell ever straddles two materials.
type Grid struct {
	NZ     int       // number of cells (layers)
	Dz     float64   // representative uniform cell thickness [m] (depth/NZ)
	Depth  float64   // total column thickness [m]
	Z      []float64 // nodal depths, length NZ, [m]
	ZFace  []float64 // interface depths, length NZ+1; ZFace[0]=0 top, ZFace[NZ]=L bottom
	DzCell []float64 // per-cell thickness [m], length NZ; nil means uniform Dz
}

// NewGrid builds a uniform cell-centred grid of nz cells over a column of
// total thickness depth [m].
func NewGrid(nz int, depth float64) Grid {
	dz := depth / float64(nz)
	g := Grid{NZ: nz, Dz: dz, Depth: depth,
		Z: make([]float64, nz), ZFace: make([]float64, nz+1)}
	for i := 0; i <= nz; i++ {
		g.ZFace[i] = float64(i) * dz
	}
	for i := 0; i < nz; i++ {
		g.Z[i] = (float64(i) + 0.5) * dz
	}
	return g
}

// LayerSpec describes one material segment's geometry for grid construction:
// Thickness [m] split into NZ equal-thickness cells. Segment interfaces are
// placed exactly on cell faces.
type LayerSpec struct {
	Thickness float64
	NZ        int
}

// layeredSumTol is the tight (but floating-point realistic) tolerance within
// which the segment thicknesses must add up to the declared column depth.
const layeredSumTol = 1e-10

// NewLayeredGrid builds a piecewise-uniform cell-centred grid from material
// segments. Segments are ordered from top to bottom, butt end to end; their
// thicknesses must sum to depth and each must carry at least one cell. The
// returned grid's faces line up exactly with every segment interface.
func NewLayeredGrid(depth float64, layers []LayerSpec) (Grid, error) {
	if len(layers) < 1 {
		return Grid{}, fmt.Errorf("at least one material layer is required")
	}
	nz := 0
	sum := 0.0
	for k, l := range layers {
		if l.Thickness <= 0 {
			return Grid{}, fmt.Errorf("layer %d thickness must be > 0, got %g", k, l.Thickness)
		}
		if l.NZ < 1 {
			return Grid{}, fmt.Errorf("layer %d must contain at least one cell", k)
		}
		nz += l.NZ
		sum += l.Thickness
	}
	tol := layeredSumTol * depth
	if tol < 1e-12 {
		tol = 1e-12
	}
	if d := sum - depth; d > tol || d < -tol {
		return Grid{}, fmt.Errorf("sum of layer thicknesses %g does not equal column depth %g", sum, depth)
	}

	g := Grid{NZ: nz, Dz: depth / float64(nz), Depth: depth,
		Z: make([]float64, nz), ZFace: make([]float64, nz+1),
		DzCell: make([]float64, nz)}
	g.ZFace[0] = 0
	base := 0.0 // cumulative thickness of finished segments
	i0 := 0
	for _, l := range layers {
		dz := l.Thickness / float64(l.NZ)
		for j := 0; j < l.NZ; j++ {
			g.DzCell[i0+j] = dz
			g.Z[i0+j] = base + (float64(j)+0.5)*dz
		}
		for j := 1; j <= l.NZ; j++ {
			g.ZFace[i0+j] = base + float64(j)*dz
		}
		// Pin the segment end face to the exact cumulative interface depth:
		// the material boundary is a geometric fact, not a rounded quotient.
		base += l.Thickness
		g.ZFace[i0+l.NZ] = base
		i0 += l.NZ
	}
	g.ZFace[nz] = depth
	return g, nil
}

// IsUniform reports whether every cell has the same thickness (bitwise), in
// which case the uniform discretisation is recovered exactly.
func (g Grid) IsUniform() bool {
	if len(g.DzCell) == 0 {
		return true
	}
	for _, d := range g.DzCell {
		if d != g.DzCell[0] {
			return false
		}
	}
	return true
}

// interblockMean selects the inter-cell hydraulic conductivity.
//
// Production computations always use the harmonic mean, as required:
// conductivity spans orders of magnitude near a wetting front and the
// arithmetic mean overestimates the inter-block flux, distorting front
// speed and direction. ArithmeticMean exists solely so the test suite can
// demonstrate the two are not interchangeable.
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

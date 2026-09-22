package solver

// Grid is the cell-centred finite-volume discretisation of the vertical
// soil column. Cells are equally thick; nodal depth z_i is the centre of
// cell i, counted positive downward from the soil surface (z = 0).
type Grid struct {
	NZ    int       // number of cells (layers)
	Dz    float64   // uniform cell thickness [m]
	Depth float64   // total column thickness [m]
	Z     []float64 // nodal depths, length NZ, [m]
	ZFace []float64 // interface depths, length NZ+1; ZFace[0]=0 top, ZFace[NZ]=L bottom
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

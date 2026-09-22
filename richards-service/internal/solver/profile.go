package solver

import "richards-service/internal/constitutive"

// Segment is one homogeneous material layer of a layered soil profile:
// everything between two successive material interfaces uses the same
// van Genuchten–Mualem constitutive curves.
type Segment struct {
	Thickness float64                  `json:"thickness_m"`
	NZ        int                      `json:"num_layers"` // equally thick cells allocated inside this segment
	Params    constitutive.Params      `json:"params"`
}

// Profile is an ordered (top to bottom) collection of material segments.
// Segments abut in depth order; their thicknesses sum to the column
// thickness. Every constitutive evaluation at a cell uses that cell's own
// segment's retention and conductivity curves.
type Profile struct {
	Segments []Segment `json:"segments"`
}

// NumSegments is the number of material segments.
func (p *Profile) NumSegments() int {
	if p == nil {
		return 1
	}
	return len(p.Segments)
}

// Depth returns the total profile thickness.
func (p *Profile) Depth() float64 {
	d := 0.0
	if p != nil {
		for _, s := range p.Segments {
			d += s.Thickness
		}
	}
	return d
}

// Thickness / SegmentCells expose the per-segment geometry used to build the
// face-aligned grid.
func (p *Profile) Thickness() []float64 {
	t := make([]float64, len(p.Segments))
	for i, s := range p.Segments {
		t[i] = s.Thickness
	}
	return t
}

func (p *Profile) SegmentCells() []int {
	n := make([]int, len(p.Segments))
	for i, s := range p.Segments {
		n[i] = s.NZ
	}
	return n
}

// cellParams expands the profile into one constitutive parameter set per
// grid cell, honouring the grid's cell-to-segment ownership. A nil profile
// yields n copies of uniform, which is the single-material case.
func cellParams(n int, materialOf []int, uniform constitutive.Params,
	prof *Profile) []constitutive.Params {
	out := make([]constitutive.Params, n)
	if prof == nil {
		for i := range out {
			out[i] = uniform
		}
		return out
	}
	for i := range out {
		out[i] = prof.Segments[materialOf[i]].Params
	}
	return out
}

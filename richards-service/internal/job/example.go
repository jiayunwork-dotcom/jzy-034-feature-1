package job

import "richards-service/internal/solver"

// SandPondingExampleID identifies the built-in verifiable scenario.
const SandPondingExampleID = "sand_ponding"

// LayeredSandExampleID identifies the layered fine-over-coarse sand
// infiltration scenario.
const LayeredSandExampleID = "layered_fine_over_coarse_sand"

// SandPondingRequest returns the preset sand-column ponded-infiltration
// example: a wetting front must move downward over time while total storage
// increases, giving a one-glance check of front direction and mass closure.
//
// Column: 1 m sand, 50 layers; alpha = 6 /m, n = 2 (m locked to 0.5),
// thetaR = 0.05, thetaS = 0.40, Ks = 5e-5 m/s. Initial uniform moisture
// theta = 0.15, 2 cm ponded head at the top, free drainage at the bottom.
// With these settings the front visibly descends through the column over a
// 90-minute run while the top cumulative inflow matches the storage gain to
// machine precision.
func SandPondingRequest() Request {
	nz := 50
	thetaInit := make([]float64, nz)
	for i := range thetaInit {
		thetaInit[i] = 0.15
	}
	m := Material{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	return Request{
		Column:   Column{Thickness: 1.0, NZ: nz},
		Material: &m,
		Initial:  Initial{Kind: "water_content", WaterContent: thetaInit},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 5400, StepSize: 30},
	}
}

// LayeredSandRequest returns the preset layered ponded-infiltration example:
// fine sand over coarse sand. The 0.4 m interface lies exactly on a grid
// face (20 + 30 cells). Under ponding the coarse layer stays unsaturated
// ahead of the front; at those tensions its K is much lower than the fine
// sand's (its larger alpha drains first), even though its Ks is ~7x higher,
// so the front ponds up at and crosses the interface at a clearly changed
// pace. Water content on the two sides is evaluated against each material's
// own bounds and need not match.
func LayeredSandRequest() Request {
	thetaInit := make([]float64, 50)
	for i := 0; i < 20; i++ {
		thetaInit[i] = 0.12 // fine sand: within [0.045, 0.43]
	}
	for i := 20; i < 50; i++ {
		thetaInit[i] = 0.05 // coarse sand: within [0.02, 0.36]
	}
	return Request{
		Column: Column{Thickness: 1.0, NZ: 50},
		Materials: []MaterialLayer{
			{Thickness: 0.4, NumLayers: 20, Alpha: 4.5, N: 2.68,
				ThetaR: 0.045, ThetaS: 0.43, Ks: 1.7e-5}, // fine sand
			{Thickness: 0.6, NumLayers: 30, Alpha: 10.0, N: 2.68,
				ThetaR: 0.02, ThetaS: 0.36, Ks: 1.2e-4}, // coarse sand
		},
		Initial:  Initial{Kind: "water_content", WaterContent: thetaInit},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 14400, StepSize: 120},
	}
}

// ExampleMetadata describes a built-in scenario.
type ExampleMetadata struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

// ListExamples returns the preset scenarios.
func ListExamples() []ExampleMetadata {
	return []ExampleMetadata{
		{
			ID:    SandPondingExampleID,
			Title: "Sand column ponded infiltration",
			Description: "1 m homogeneous sand column (alpha=6 /m, n=2, thetaR=0.05, " +
				"thetaS=0.40, Ks=5e-5 m/s), uniform initial moisture 0.15, " +
				"2 cm ponded head at the top, free drainage at the bottom. " +
				"The wetting front moves downward and storage rises.",
		},
		{
			ID:    LayeredSandExampleID,
			Title: "Layered fine-over-coarse sand ponded infiltration",
			Description: "0.4 m fine sand over 0.6 m coarse sand (interface on a grid " +
				"face); each segment carries its own van Genuchten-Mualem curve. " +
				"Under unsaturated conditions the coarse layer's conductivity is " +
				"well below the fine layer's, so the wetting-front speed changes " +
				"observably as the front reaches and crosses the interface; the " +
				"interface flux stays continuous and the column budget closes.",
		},
	}
}

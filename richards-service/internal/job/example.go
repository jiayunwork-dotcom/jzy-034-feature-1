package job

import "richards-service/internal/solver"

// SandPondingExampleID identifies the built-in verifiable scenario.
const SandPondingExampleID = "sand_ponding"

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
	return Request{
		Column:   Column{Thickness: 1.0, NZ: nz},
		Material: Material{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5},
		Initial:  Initial{Kind: "water_content", WaterContent: thetaInit},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 5400, StepSize: 30},
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
	return []ExampleMetadata{{
		ID:    SandPondingExampleID,
		Title: "Sand column ponded infiltration",
		Description: "1 m sand column (alpha=6 /m, n=2, thetaR=0.05, " +
			"thetaS=0.40, Ks=5e-5 m/s), uniform initial moisture 0.15, " +
			"2 cm ponded head at the top, free drainage at the bottom. " +
			"The wetting front moves downward and storage rises.",
	}}
}

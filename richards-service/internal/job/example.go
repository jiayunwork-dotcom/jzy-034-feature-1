package job

import "richards-service/internal/solver"

// SandPondingExampleID identifies the built-in verifiable scenario.
const SandPondingExampleID = "sand_ponding"

// LayeredSandExampleID identifies the built-in layered-profile scenario.
const LayeredSandExampleID = "layered_sand_over_coarse"

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

// LayeredSandRequest returns a two-segment ponded-infiltration example:
// 0.4 m fine sand over 0.6 m coarse sand, each segment with its own
// van Genuchten–Mualem parameters. The coarse sublayer is less conductive
// than the fine top layer at the same (unsaturated) pressure head, so the
// wetting front visibly hesitates at the material interface before
// breaking through — the canonical capillary-barrier behaviour.
//
// Fine sand:  alpha=7 /m, n=2.2, thetaR=0.05, thetaS=0.41, Ks=6e-5 m/s.
// Coarse sand: alpha=14 /m, n=2.6, thetaR=0.04, thetaS=0.43, Ks=1.2e-4 m/s.
// Initial uniform moisture 0.12, 2 cm ponded head, free drainage.
func LayeredSandRequest() Request {
	return Request{
		Column: Column{Thickness: 1.0, NZ: 60},
		Profile: &SoilProfileRequest{Layers: []SoilLayer{
			{Thickness: 0.4, NumLayers: 24,
				Material: Material{Alpha: 7.0, N: 2.2, ThetaR: 0.05, ThetaS: 0.41, Ks: 6e-5}},
			{Thickness: 0.6, NumLayers: 36,
				Material: Material{Alpha: 14.0, N: 2.6, ThetaR: 0.04, ThetaS: 0.43, Ks: 1.2e-4}},
		}},
		Initial:  Initial{Kind: "water_content", WaterContent: uniformValues(0.12, 60)},
		Boundary: Boundary{Top: solver.TopPondedHead, PondedHead: 0.02, Bottom: solver.BottomFreeDrainage},
		Time:     TimeSpec{TotalTime: 10800, StepSize: 60},
	}
}

func uniformValues(v float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
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
			Description: "1 m sand column (alpha=6 /m, n=2, thetaR=0.05, " +
				"thetaS=0.40, Ks=5e-5 m/s), uniform initial moisture 0.15, " +
				"2 cm ponded head at the top, free drainage at the bottom. " +
				"The wetting front moves downward and storage rises.",
		},
		{
			ID:    LayeredSandExampleID,
			Title: "Layered fine-over-coarse sand ponded infiltration",
			Description: "1 m column: 0.4 m fine sand over 0.6 m coarse sand, " +
				"each segment with its own van Genuchten-Mualem parameters. " +
				"The wetting front slows at the material interface (capillary " +
				"barrier) before advancing into the coarse sublayer.",
		},
	}
}

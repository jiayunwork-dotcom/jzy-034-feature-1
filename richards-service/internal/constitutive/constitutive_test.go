package constitutive

import (
	"math"
	"testing"
)

func TestMLockedToN(t *testing.T) {
	p := Params{Alpha: 2, N: 2.68, ThetaR: 0.05, ThetaS: 0.4, Ks: 1e-5}
	want := 1.0 - 1.0/2.68
	if math.Abs(p.M()-want) > 1e-15 {
		t.Fatalf("m binding wrong: got %.16f want %.16f", p.M(), want)
	}
}

func TestEffectiveSaturationEndpoints(t *testing.T) {
	p := Params{Alpha: 6, N: 2, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	if got := p.EffectiveSaturation(0); got != 1 {
		t.Fatalf("Se(0) = %v, want 1", got)
	}
	if got := p.EffectiveSaturation(0.37); got != 1 {
		t.Fatalf("Se(positive) = %v, want 1", got)
	}
	if got := p.EffectiveSaturation(-1e9); got > 1e-6 {
		t.Fatalf("Se(very dry) = %v, want ~0", got)
	}
	// spot value of the closed form
	h := -0.1
	x := 6.0 * 0.1
	want := math.Pow(1+math.Pow(x, 2), -0.5)
	if got := p.EffectiveSaturation(h); math.Abs(got-want) > 1e-12 {
		t.Fatalf("Se(-0.1) = %.10f want %.10f", got, want)
	}
}

func TestWaterContentBounds(t *testing.T) {
	p := Params{Alpha: 6, N: 2, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	for _, h := range []float64{-100, -1, -0.01, 0, 0.5} {
		th := p.WaterContent(h)
		if th < p.ThetaR-1e-15 || th > p.ThetaS+1e-15 {
			t.Fatalf("theta(%g)=%g out of range", h, th)
		}
	}
}

func TestHeadFromWaterContentRoundTrip(t *testing.T) {
	p := Params{Alpha: 6, N: 2, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	for _, th := range []float64{0.06, 0.1, 0.2, 0.35} {
		h := p.HeadFromWaterContent(th)
		back := p.WaterContent(h)
		if math.Abs(back-th) > 1e-10 {
			t.Fatalf("round trip theta=%g h=%g back=%g", th, h, back)
		}
	}
	if got := p.HeadFromWaterContent(p.ThetaS); got != 0 {
		t.Fatalf("inverse at thetaS: h=%v want 0", got)
	}
}

func TestRelativePermeabilityMonotonic(t *testing.T) {
	m := 0.5
	prev := 0.0
	for i := 1; i <= 100; i++ {
		se := float64(i) / 100.0
		kr := RelativePermeability(se, m)
		if kr < prev-1e-15 || kr < 0 || kr > 1+1e-12 {
			t.Fatalf("Kr not monotone/bounded at Se=%g: %g", se, kr)
		}
		prev = kr
	}
	if RelativePermeability(0, m) != 0 {
		t.Fatal("Kr(0) should be 0")
	}
	if RelativePermeability(1, m) != 1 {
		t.Fatal("Kr(1) should be 1")
	}
}

func TestCapacityConsistentWithSmoothing(t *testing.T) {
	p := Params{Alpha: 6, N: 2, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	// Across the smoothing band, analytic dtheta/dh must match central
	// finite differences of the actual theta(h). Avoid the extreme tip
	// where 1-Se is at the machine-epsilon scale for the FD step.
	for _, h := range []float64{-0.0019, -0.0015, -0.001, -7e-4, -4e-4} {
		const eps = 1e-9
		num := (p.WaterContent(h+eps) - p.WaterContent(h-eps)) / (2 * eps)
		ana := p.DThetaDH(h)
		scale := math.Max(math.Abs(num), 1e-8)
		if math.Abs(num-ana)/scale > 2e-4 {
			t.Fatalf("C mismatch at h=%g ana=%.6e num=%.6e", h, ana, num)
		}
	}
	// Saturated branch capacity is zero and the curve stays at thetaS.
	if p.DThetaDH(0.1) != 0 {
		t.Fatal("saturated capacity should be zero")
	}
	if p.WaterContent(-1e-12) > p.ThetaS {
		t.Fatal("theta cannot exceed thetaS")
	}
}

func TestDKDhFiniteDifference(t *testing.T) {
	p := Params{Alpha: 6, N: 2, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
	for _, h := range []float64{-1.0, -0.1, -0.01, -0.001} {
		const eps = 1e-7
		num := (p.Conductivity(h+eps) - p.Conductivity(h-eps)) / (2 * eps)
		ana := p.DKDh(h)
		scale := math.Max(math.Abs(num), 1e-12)
		if math.Abs(num-ana)/scale > 1e-5 {
			t.Fatalf("dK/dh mismatch at h=%g ana=%.6e num=%.6e", h, ana, num)
		}
	}
}

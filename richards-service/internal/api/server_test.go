package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"richards-service/internal/job"
)

func testRouter() http.Handler {
	return NewServer(job.NewManager())
}

func doJSON(t *testing.T, h http.Handler, method, path string, body any) (int, map[string]any) {
	t.Helper()
	var rdr *bytes.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		rdr = bytes.NewReader(b)
	} else {
		rdr = bytes.NewReader(nil)
	}
	req := httptest.NewRequest(method, path, rdr)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	out := map[string]any{}
	if w.Body.Len() > 0 {
		if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
			t.Fatalf("invalid JSON (%v): %s", err, w.Body.String())
		}
	}
	return w.Code, out
}

func goodJobBody() job.Request {
	return job.SandPondingRequest()
}

func TestHealthAndStatus(t *testing.T) {
	h := testRouter()
	if code, body := doJSON(t, h, "GET", "/healthz", nil); code != http.StatusOK || body["status"] != "ok" {
		t.Fatalf("health %d %v", code, body)
	}
	code, body := doJSON(t, h, "GET", "/status", nil)
	if code != http.StatusOK {
		t.Fatalf("status %d", code)
	}
	if _, ok := body["jobs_submitted"]; !ok {
		t.Fatal("status missing counters")
	}
}

func TestConstitutiveEcho(t *testing.T) {
	h := testRouter()
	code, body := doJSON(t, h, "GET", "/api/v1/constitutive", nil)
	if code != http.StatusOK {
		t.Fatalf("code %d", code)
	}
	eqs, _ := body["equations"].(map[string]any)
	for _, key := range []string{"effective_saturation", "water_content", "conductivity", "interblock_k"} {
		if _, ok := eqs[key]; !ok {
			t.Fatalf("missing equation %s", key)
		}
	}
	cstr, _ := body["constants"].(map[string]any)
	if cstr["l_mualem"].(float64) != 0.5 {
		t.Fatalf("l = %v", cstr["l_mualem"])
	}
	if !strings.Contains(eqs["m_binding"].(string), "1 - 1/n") {
		t.Fatalf("m binding echo: %v", eqs["m_binding"])
	}
}

func TestSubmitFullJob(t *testing.T) {
	h := testRouter()
	code, body := doJSON(t, h, "POST", "/api/v1/jobs", goodJobBody())
	if code != http.StatusOK {
		t.Fatalf("code %d body=%v", code, body)
	}
	if body["job_id"] == nil || body["job_id"] == "" {
		t.Fatal("missing job_id")
	}
	res, _ := body["result"].(map[string]any)
	steps, _ := res["steps"].([]any)
	if len(steps) == 0 {
		t.Fatal("no steps returned")
	}
	last := steps[len(steps)-1].(map[string]any)
	layers, _ := last["layers"].([]any)
	if len(layers) != 50 {
		t.Fatalf("layers=%d", len(layers))
	}
	if res["total_mass_balance_residual_m"].(float64) > 1e-8 {
		t.Fatalf("closure %v", res["total_mass_balance_residual_m"])
	}
}

func TestSubmitSingleStep(t *testing.T) {
	h := testRouter()
	req := job.SandPondingRequest()
	body := job.StepRequest{
		Column:   req.Column,
		Material: req.Material,
		Initial:  req.Initial,
		Boundary: req.Boundary,
		StepSize: req.Time.StepSize,
	}
	code, out := doJSON(t, h, "POST", "/api/v1/steps", body)
	if code != http.StatusOK {
		t.Fatalf("code %d %v", code, out)
	}
	res := out["result"].(map[string]any)
	if res["before"] == nil || res["after"] == nil {
		t.Fatal("single step must return before/after")
	}
	step := res["step"].(map[string]any)
	if step["mass_balance_residual_m"].(float64) > 1e-8 {
		t.Fatalf("closure %v", step["mass_balance_residual_m"])
	}
}

func TestIllegalParameterStructuredError(t *testing.T) {
	h := testRouter()
	bad := goodJobBody()
	bad.Material.N = 1.0
	code, body := doJSON(t, h, "POST", "/api/v1/jobs", bad)
	if code != http.StatusUnprocessableEntity {
		t.Fatalf("code %d", code)
	}
	e, _ := body["error"].(map[string]any)
	if e["type"] != "VALIDATION_ERROR" || e["code"] != "N_NOT_GREATER_THAN_ONE" {
		t.Fatalf("error body %v", e)
	}
}

func TestAllFiveIllegalCodesOverHTTP(t *testing.T) {
	cases := []struct {
		mut  func(r *job.Request)
		code string
	}{
		{func(r *job.Request) { r.Material.N = 1 }, "N_NOT_GREATER_THAN_ONE"},
		{func(r *job.Request) { r.Material.Alpha = -1 }, "ALPHA_NON_POSITIVE"},
		{func(r *job.Request) { r.Material.ThetaR = r.Material.ThetaS }, "THETA_R_GE_THETA_S"},
		{func(r *job.Request) { r.Material.Ks = 0 }, "KS_NON_POSITIVE"},
		{func(r *job.Request) { r.Column.Thickness = 0 }, "COLUMN_THICKNESS_NON_POSITIVE"},
	}
	for _, tc := range cases {
		h := testRouter()
		b := goodJobBody()
		tc.mut(&b)
		code, body := doJSON(t, h, "POST", "/api/v1/jobs", b)
		if code != http.StatusUnprocessableEntity {
			t.Fatalf("%s: http code %d", tc.code, code)
		}
		e := body["error"].(map[string]any)
		if e["code"] != tc.code {
			t.Fatalf("got %s want %s", e["code"], tc.code)
		}
	}
}

func TestMalformedJSON(t *testing.T) {
	h := testRouter()
	req := httptest.NewRequest("POST", "/api/v1/jobs", strings.NewReader("{not json"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("code %d", w.Code)
	}
}

func TestExampleEndpoints(t *testing.T) {
	h := testRouter()
	code, body := doJSON(t, h, "GET", "/api/v1/examples", nil)
	if code != http.StatusOK {
		t.Fatalf("code %d", code)
	}
	lst := body["examples"].([]any)
	if len(lst) != 1 {
		t.Fatalf("examples %d", len(lst))
	}
	code, body = doJSON(t, h, "GET", "/api/v1/examples/sand-ponding", nil)
	if code != http.StatusOK || body["id"] != job.SandPondingExampleID {
		t.Fatalf("example endpoint %d %v", code, body)
	}
}

func TestExampleSubmitsDirectly(t *testing.T) {
	h := testRouter()
	_, body := doJSON(t, h, "GET", "/api/v1/examples/sand-ponding", nil)
	req := body["request"]
	code, out := doJSON(t, h, "POST", "/api/v1/jobs", req)
	if code != http.StatusOK {
		t.Fatalf("submit preset example: %d %v", code, out)
	}
}

func TestConcurrentHTTPSubmissionsIsolated(t *testing.T) {
	gin.SetMode(gin.ReleaseMode)
	h := testRouter()
	done := make(chan map[string]any, 16)
	for j := 0; j < 16; j++ {
		go func() {
			b := goodJobBody()
			b.Time.TotalTime = 300
			code, out := doJSON(t, h, "POST", "/api/v1/jobs", b)
			if code != http.StatusOK {
				t.Errorf("concurrent submit code %d", code)
				done <- nil
				return
			}
			done <- out
		}()
	}
	ids := map[string]bool{}
	for j := 0; j < 16; j++ {
		out := <-done
		if out == nil {
			continue
		}
		id := out["job_id"].(string)
		if ids[id] {
			t.Fatal("duplicate job id")
		}
		ids[id] = true
	}
}

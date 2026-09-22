// Package api wires the HTTP surface (Gin) to the job orchestrator.
package api

import (
	"errors"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"richards-service/internal/constitutive"
	"richards-service/internal/job"
	"richards-service/internal/solver"
)

// Service version / constitutive constants echoed by the meta endpoint.
const (
	ServiceName    = "richards-seepage-service"
	ServiceVersion = "1.0.0"
)

// Server holds HTTP dependencies.
type Server struct {
	manager *job.Manager
	started time.Time
}

// NewServer constructs the Gin engine with all routes registered.
func NewServer(m *job.Manager) *gin.Engine {
	srv := &Server{manager: m, started: time.Now()}
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(requestID())

	v1 := r.Group("/api/v1")
	{
		v1.POST("/jobs", srv.submitJob)           // full infiltration run
		v1.POST("/steps", srv.submitSingleStep)   // one implicit time step
		v1.GET("/constitutive", srv.constitutive) // read-only model form
		v1.GET("/examples", srv.listExamples)
		v1.GET("/examples/sand-ponding", srv.sandExample)
		v1.GET("/examples/layered-sand", srv.layeredExample)
	}
	r.GET("/healthz", srv.health)
	r.GET("/status", srv.status) // basic runtime state for monitoring
	return r
}

func requestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("X-Service", ServiceName)
		c.Next()
	}
}

// apiError is the typed structured error body.
type apiError struct {
	Error struct {
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
		Field   string `json:"field,omitempty"`
		Message string `json:"message"`
	} `json:"error"`
}

func fail(c *gin.Context, status int, typ, code, field, msg string) {
	var e apiError
	e.Error.Type = typ
	e.Error.Code = code
	e.Error.Field = field
	e.Error.Message = msg
	c.AbortWithStatusJSON(status, e)
}

func (s *Server) submitJob(c *gin.Context) {
	var req job.Request
	if err := c.ShouldBindJSON(&req); err != nil {
		fail(c, http.StatusBadRequest, "INVALID_JSON", "", "", err.Error())
		return
	}
	id, res, err := s.manager.RunFull(req)
	if err != nil {
		writeRunError(c, err)
		return
	}
	c.JSON(http.StatusOK, map[string]any{"job_id": id, "result": res})
}

func (s *Server) submitSingleStep(c *gin.Context) {
	var req job.StepRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		fail(c, http.StatusBadRequest, "INVALID_JSON", "", "", err.Error())
		return
	}
	id, res, err := s.manager.RunSingle(req)
	if err != nil {
		writeRunError(c, err)
		return
	}
	c.JSON(http.StatusOK, map[string]any{"job_id": id, "result": res})
}

func writeRunError(c *gin.Context, err error) {
	var ve *job.ValidationError
	if errors.As(err, &ve) {
		fail(c, http.StatusUnprocessableEntity, "VALIDATION_ERROR",
			ve.Code, ve.Field, ve.Message)
		return
	}
	var fe *solver.Failure
	if errors.As(err, &fe) {
		status := http.StatusUnprocessableEntity
		if fe.Kind == solver.FailNonFinite {
			status = http.StatusInternalServerError
		}
		fail(c, status, "COMPUTATION_FAILURE", fe.Kind, "", fe.Message)
		return
	}
	fail(c, http.StatusInternalServerError, "INTERNAL_ERROR", "", "", err.Error())
}

type constitutiveResponse struct {
	Equations map[string]string  `json:"equations"`
	Constants map[string]float64 `json:"constants"`
	Params    []string           `json:"parameter_order"`
	Notes     []string           `json:"notes"`
}

func (s *Server) constitutive(c *gin.Context) {
	resp := constitutiveResponse{
		Equations: map[string]string{
			"effective_saturation": "Se = [1 + (alpha*|h|)^n]^(-m) for h < 0; Se = 1 for h >= 0",
			"water_content":        "theta = theta_r + (theta_s - theta_r) * Se",
			"conductivity":         "K = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2",
			"richards_equation":    "dtheta/dt = d/dz [ K(h) * (dh/dz - 1) ], z positive downward",
			"time_scheme":          "fully implicit backward Euler; full-Newton iteration (exact Jacobian, backtracking line search) per step; adaptive internal substepping",
			"interblock_k":         "Kf = 2*K_i*K_{i+1}/(K_i+K_{i+1)} (harmonic mean, equal spacing)",
			"interface_k":          "layered profile: Kf = (lu+ld)*Ku*Kd/(ld*Ku + lu*Kd) at a material interface (distance-weighted harmonic mean; flux continuous, h and theta may jump)",
		},
		Constants: map[string]float64{
			"m":                      -1, // locked, echoed symbolically below
			"l_mualem":               constitutive.PoresConnectivityL,
			"n_lower_bound":          1.0,
			"head_tolerance_m":       solver.DefaultOptions().HeadTol,
			"residual_tolerance_1_s": solver.DefaultOptions().ResidualTol,
			"max_iterations":         float64(solver.DefaultOptions().MaxIterations),
		},
		Params: []string{"alpha", "n", "theta_r", "theta_s", "ks"},
		Notes: []string{
			"m is locked to m = 1 - 1/n",
			"Mualem pore-connectivity exponent l is fixed at 0.5",
			"units: m, s, K in m/s; flux positive downward",
			"inter-cell conductivity uses the harmonic mean (never arithmetic)",
			"bottom boundary is fixed per job: free_drainage or zero_flux",
			"a job carries either one material block (uniform column) or a profile of material segments (layered column); every cell uses its own segment's retention and conductivity curves",
			"material interfaces coincide exactly with cell faces; per-segment parameter legality (n>1, alpha>0, theta_r<theta_s, ks>0) is enforced per segment",
			"the constitutive structure actually used by a job is echoed in its result under material_config",
		},
	}
	// m is not a constant number; present the binding as an equation and
	// remove the placeholder.
	delete(resp.Constants, "m")
	resp.Equations["m_binding"] = "m = 1 - 1/n"
	c.JSON(http.StatusOK, resp)
}

func (s *Server) listExamples(c *gin.Context) {
	c.JSON(http.StatusOK, map[string]any{"examples": job.ListExamples()})
}

func (s *Server) sandExample(c *gin.Context) {
	c.JSON(http.StatusOK, map[string]any{
		"id":      job.SandPondingExampleID,
		"request": job.SandPondingRequest(),
	})
}

func (s *Server) layeredExample(c *gin.Context) {
	c.JSON(http.StatusOK, map[string]any{
		"id":      job.LayeredSandExampleID,
		"request": job.LayeredSandRequest(),
	})
}

func (s *Server) health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": ServiceName,
		"version": ServiceVersion,
	})
}

func (s *Server) status(c *gin.Context) {
	st := s.manager.StatsSnapshot()
	c.JSON(http.StatusOK, gin.H{
		"service":           ServiceName,
		"version":           ServiceVersion,
		"uptime_s":          time.Since(s.started).Seconds(),
		"jobs_submitted":    st.Submitted,
		"jobs_completed":    st.Completed,
		"jobs_failed":       st.Failed,
		"jobs_in_flight":    st.InFlight,
		"steps_accepted":    st.StepsAccepted,
		"started_unix_nano": st.StartedUnixNano,
	})
}

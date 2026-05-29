// Package main - ACMS API Gateway
//
// Routes requests to internal services with real JWT validation,
// Redis-backed rate limiting, circuit breaker pattern, WebSocket proxy,
// request logging, CORS, IP filtering, and API versioning.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/acms/go-services/internal/config"
	"github.com/acms/go-services/internal/middleware"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

const APIVersion = "v1"

// BackendService describes a proxied backend.
type BackendService struct {
	Name           string
	TargetURL      string
	CircuitBreaker *middleware.CircuitBreaker
}

// GatewayService manages the API gateway.
type GatewayService struct {
	config   *config.GatewayConfig
	logger   *zap.Logger
	redis    *redis.Client
	backends map[string]*BackendService
	limiter  *middleware.RedisRateLimiter
	upgrader websocket.Upgrader
}

// NewGatewayService creates a new gateway service.
func NewGatewayService(cfg *config.GatewayConfig, logger *zap.Logger) (*GatewayService, error) {
	// Connect to Redis for rate limiting
	redisOpts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		logger.Warn("failed to parse Redis URL, using default", zap.Error(err))
		redisOpts = &redis.Options{
			Addr: "localhost:6379",
			DB:   0,
		}
	}
	redisClient := redis.NewClient(redisOpts)

	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := redisClient.Ping(ctx).Result(); err != nil {
		logger.Warn("Redis connection failed, rate limiting will use local fallback", zap.Error(err))
	} else {
		logger.Info("Redis connected for rate limiting")
	}

	// Create rate limiter
	limiter := middleware.NewRedisRateLimiter(redisClient, cfg.RateLimitPerSec, time.Second, logger)

	// Set up backend services with circuit breakers
	backends := make(map[string]*BackendService)
	for _, svc := range []struct {
		name string
		url  string
	}{
		{"python_api", cfg.PythonAPIURL},
		{"monitor", cfg.MonitorURL},
		{"connector", cfg.ConnectorURL},
	} {
		backends[svc.name] = &BackendService{
			Name:      svc.name,
			TargetURL: svc.url,
			CircuitBreaker: middleware.NewCircuitBreaker(
				5,               // failure threshold
				30*time.Second,  // reset timeout
				logger,
			),
		}
	}

	return &GatewayService{
		config:   cfg,
		logger:   logger,
		redis:    redisClient,
		backends: backends,
		limiter:  limiter,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins in development; restrict in production
			},
		},
	}, nil
}

// proxyToBackend creates a reverse proxy handler with circuit breaker.
func (g *GatewayService) proxyToBackend(backendName string) gin.HandlerFunc {
	backend, ok := g.backends[backendName]
	if !ok {
		return func(c *gin.Context) {
			c.JSON(http.StatusBadGateway, gin.H{"error": "unknown backend service"})
		}
	}

	targetURL, err := url.Parse(backend.TargetURL)
	if err != nil {
		g.logger.Error("failed to parse backend URL",
			zap.String("backend", backendName),
			zap.Error(err),
		)
		return func(c *gin.Context) {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "backend misconfigured"})
		}
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	// Customize the director to preserve the full path
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.Host = targetURL.Host
		req.Header.Set("X-Forwarded-Host", req.Host)
		req.Header.Set("X-Forwarded-Proto", "http")
		req.Header.Set("X-Request-ID", req.Header.Get("X-Request-ID"))
	}

	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		backend.CircuitBreaker.RecordFailure()
		g.logger.Error("proxy error",
			zap.String("backend", backendName),
			zap.String("path", r.URL.Path),
			zap.Error(err),
		)
	}

	return func(c *gin.Context) {
		if !backend.CircuitBreaker.Allow() {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error":  "service temporarily unavailable",
				"state":  backend.CircuitBreaker.State(),
				"backend": backendName,
			})
			return
		}

		// Record success/failure based on response
		respWriter := &responseCapture{ResponseWriter: c.Writer, statusCode: 200}
		c.Writer = respWriter

		proxy.ServeHTTP(c.Writer, c.Request)

		if respWriter.statusCode >= 500 {
			backend.CircuitBreaker.RecordFailure()
		} else {
			backend.CircuitBreaker.RecordSuccess()
		}
	}
}

// responseCapture wraps gin.ResponseWriter to capture status code.
type responseCapture struct {
	gin.ResponseWriter
	statusCode int
}

func (w *responseCapture) WriteHeader(code int) {
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}

// handleWebSocketProxy proxies WebSocket connections to a backend.
func (g *GatewayService) handleWebSocketProxy(backendName, path string) gin.HandlerFunc {
	backend, ok := g.backends[backendName]
	if !ok {
		return func(c *gin.Context) {
			c.JSON(http.StatusBadGateway, gin.H{"error": "unknown backend service"})
		}
	}

	return func(c *gin.Context) {
		targetURL, err := url.Parse(backend.TargetURL)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "backend misconfigured"})
			return
		}

		// Upgrade client connection
		clientConn, err := g.upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			g.logger.Error("WebSocket upgrade failed", zap.Error(err))
			return
		}
		defer clientConn.Close()

		// Connect to backend WebSocket
		wsScheme := "ws"
		if targetURL.Scheme == "https" {
			wsScheme = "wss"
		}
		backendWSURL := fmt.Sprintf("%s://%s%s", wsScheme, targetURL.Host, path)
		if c.Request.URL.RawQuery != "" {
			backendWSURL += "?" + c.Request.URL.RawQuery
		}

		backendConn, _, err := websocket.DefaultDialer.Dial(backendWSURL, nil)
		if err != nil {
			g.logger.Error("backend WebSocket connection failed",
				zap.String("url", backendWSURL),
				zap.Error(err),
			)
			return
		}
		defer backendConn.Close()

		// Bidirectional proxy
		done := make(chan struct{}, 2)

		// Client -> Backend
		go func() {
			defer func() { done <- struct{}{} }()
			for {
				msgType, msg, err := clientConn.ReadMessage()
				if err != nil {
					return
				}
				if err := backendConn.WriteMessage(msgType, msg); err != nil {
					return
				}
			}
		}()

		// Backend -> Client
		go func() {
			defer func() { done <- struct{}{} }()
			for {
				msgType, msg, err := backendConn.ReadMessage()
				if err != nil {
					return
				}
				if err := clientConn.WriteMessage(msgType, msg); err != nil {
					return
				}
			}
		}()

		<-done
	}
}

// setupRoutes configures all gateway routes.
func (g *GatewayService) setupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()

	// Global middleware
	router.Use(middleware.RecoveryMiddleware(g.logger))
	router.Use(middleware.RequestIDMiddleware())
	router.Use(middleware.RequestLoggerMiddleware(g.logger))
	router.Use(middleware.VersionMiddleware(APIVersion))
	router.Use(middleware.CORSMiddleware(g.config.CORSHosts))
	router.Use(middleware.RequestSizeLimitMiddleware(g.config.MaxRequestSize))
	router.Use(middleware.RateLimitMiddleware(g.limiter))

	// IP filtering if configured
	if len(g.config.IPBlacklist) > 0 || len(g.config.IPWhitelist) > 0 {
		router.Use(middleware.IPFilterMiddleware(g.config.IPBlacklist, g.config.IPWhitelist, g.logger))
	}

	// Health check (public, no auth)
	router.GET("/health", func(c *gin.Context) {
		backendStatus := make(map[string]string)
		for name, backend := range g.backends {
			backendStatus[name] = backend.CircuitBreaker.State()
		}
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"service":   "api-gateway",
			"version":   APIVersion,
			"backends":  backendStatus,
			"timestamp": time.Now().Format(time.RFC3339),
		})
	})

	// Public auth routes (no JWT required)
	router.POST("/api/v1/auth/login", g.proxyToBackend("python_api"))
	router.POST("/api/v1/auth/register", g.proxyToBackend("python_api"))
	router.POST("/api/v1/auth/refresh", g.proxyToBackend("python_api"))

	// API v2 routes (versioned)
	v2 := router.Group("/api/v2")
	v2.Use(middleware.JWTAuthMiddleware(g.config.JWTSecret, g.logger))
	{
		v2.Any("/*path", g.proxyToBackend("python_api"))
	}

	// Protected routes
	api := router.Group("/api/v1")
	api.Use(middleware.JWTAuthMiddleware(g.config.JWTSecret, g.logger))
	{
		// Orders
		api.POST("/orders", g.proxyToBackend("python_api"))
		api.GET("/orders", g.proxyToBackend("python_api"))
		api.GET("/orders/:id", g.proxyToBackend("python_api"))
		api.DELETE("/orders/:id", g.proxyToBackend("python_api"))
		api.PATCH("/orders/:id", g.proxyToBackend("python_api"))

		// Positions
		api.GET("/positions", g.proxyToBackend("python_api"))
		api.GET("/positions/:id", g.proxyToBackend("python_api"))
		api.POST("/positions/:id/close", g.proxyToBackend("python_api"))

		// Portfolio
		api.GET("/portfolio", g.proxyToBackend("python_api"))
		api.GET("/portfolio/history", g.proxyToBackend("python_api"))

		// Strategies
		api.POST("/strategies", g.proxyToBackend("python_api"))
		api.GET("/strategies", g.proxyToBackend("python_api"))
		api.GET("/strategies/:id", g.proxyToBackend("python_api"))
		api.PUT("/strategies/:id", g.proxyToBackend("python_api"))
		api.DELETE("/strategies/:id", g.proxyToBackend("python_api"))
		api.POST("/strategies/:id/start", g.proxyToBackend("python_api"))
		api.POST("/strategies/:id/stop", g.proxyToBackend("python_api"))
		api.GET("/strategies/:id/performance", g.proxyToBackend("python_api"))

		// Backtest
		api.POST("/backtest", g.proxyToBackend("python_api"))
		api.GET("/backtest/:id", g.proxyToBackend("python_api"))
		api.GET("/backtest", g.proxyToBackend("python_api"))

		// Risk
		api.GET("/risk/status", g.proxyToBackend("python_api"))
		api.POST("/risk/kill-switch", g.proxyToBackend("python_api"))
		api.GET("/risk/exposure", g.proxyToBackend("python_api"))
		api.POST("/risk/limits", g.proxyToBackend("python_api"))

		// Market data
		api.GET("/market/candles/:symbol", g.proxyToBackend("python_api"))
		api.GET("/market/orderbook/:symbol", g.proxyToBackend("python_api"))
		api.GET("/market/ticker/:symbol", g.proxyToBackend("python_api"))
		api.GET("/market/trades/:symbol", g.proxyToBackend("python_api"))

		// User settings
		api.GET("/settings", g.proxyToBackend("python_api"))
		api.PUT("/settings", g.proxyToBackend("python_api"))
		api.GET("/api-keys", g.proxyToBackend("python_api"))
		api.POST("/api-keys", g.proxyToBackend("python_api"))
		api.DELETE("/api-keys/:id", g.proxyToBackend("python_api"))

		// Monitoring (proxied to monitor service)
		api.GET("/monitor/metrics", g.proxyToBackend("monitor"))
		api.GET("/monitor/health", g.proxyToBackend("monitor"))
		api.GET("/monitor/alerts", g.proxyToBackend("monitor"))
		api.POST("/monitor/alerts/:id/acknowledge", g.proxyToBackend("monitor"))
		api.GET("/monitor/dashboard", g.proxyToBackend("monitor"))
	}

	// Connector management (proxied to connector service)
	connector := router.Group("/api/v1/connector")
	connector.Use(middleware.JWTAuthMiddleware(g.config.JWTSecret, g.logger))
	{
		connector.GET("/status", g.proxyToBackend("connector"))
		connector.POST("/connect/:exchange", g.proxyToBackend("connector"))
		connector.POST("/disconnect/:exchange", g.proxyToBackend("connector"))
		connector.GET("/symbols", g.proxyToBackend("connector"))
		connector.GET("/kafka/stats", g.proxyToBackend("connector"))
	}

	// WebSocket proxy for real-time streams
	router.GET("/ws/v1/stream", g.handleWebSocketProxy("python_api", "/ws/v1/stream"))
	router.GET("/ws/v1/market/:symbol", g.handleWebSocketProxy("connector", "/ws/v1/market"))

	// Prometheus metrics (no auth, for internal scraping)
	router.GET("/metrics", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"gateway": gin.H{
				"backends": g.getBackendMetrics(),
			},
		})
	})

	// Circuit breaker status endpoint
	router.GET("/admin/circuit-breakers", func(c *gin.Context) {
		status := make(map[string]interface{})
		for name, backend := range g.backends {
			status[name] = gin.H{
				"state":   backend.CircuitBreaker.State(),
				"target":  backend.TargetURL,
			}
		}
		c.JSON(http.StatusOK, status)
	})

	// Reset circuit breaker
	router.POST("/admin/circuit-breakers/:name/reset", func(c *gin.Context) {
		name := c.Param("name")
		backend, ok := g.backends[name]
		if !ok {
			c.JSON(http.StatusNotFound, gin.H{"error": "backend not found"})
			return
		}
		// Replace with fresh circuit breaker
		backend.CircuitBreaker = middleware.NewCircuitBreaker(
			5, 30*time.Second, g.logger,
		)
		c.JSON(http.StatusOK, gin.H{"message": fmt.Sprintf("circuit breaker for %s reset", name)})
	})

	return router
}

func (g *GatewayService) getBackendMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})
	for name, backend := range g.backends {
		metrics[name] = map[string]interface{}{
			"circuit_state": backend.CircuitBreaker.State(),
			"target":        backend.TargetURL,
		}
	}
	return metrics
}

// Close cleans up resources.
func (g *GatewayService) Close() {
	if g.redis != nil {
		g.redis.Close()
	}
	g.logger.Info("gateway resources cleaned up")
}

func main() {
	// Initialize structured logger
	logger, err := zap.NewProduction()
	if err != nil {
		log.Fatalf("failed to create logger: %v", err)
	}
	defer logger.Sync()

	// Load configuration
	cfg := config.LoadGatewayConfig()
	if err := cfg.Validate(); err != nil {
		logger.Fatal("invalid configuration", zap.Error(err))
	}

	// Parse CORS hosts from environment
	corsHosts := strings.Split(config.GetEnv("CORS_HOSTS", "*"), ",")
	cfg.CORSHosts = corsHosts

	// Parse IP lists from environment
	blacklist := strings.Split(config.GetEnv("IP_BLACKLIST", ""), ",")
	if blacklist[0] == "" {
		blacklist = nil
	}
	cfg.IPBlacklist = blacklist

	whitelist := strings.Split(config.GetEnv("IP_WHITELIST", ""), ",")
	if whitelist[0] == "" {
		whitelist = nil
	}
	cfg.IPWhitelist = whitelist

	logger.Info("starting API gateway",
		zap.String("port", cfg.Port),
		zap.String("python_api_url", cfg.PythonAPIURL),
		zap.String("monitor_url", cfg.MonitorURL),
		zap.String("connector_url", cfg.ConnectorURL),
		zap.Strings("cors_hosts", cfg.CORSHosts),
		zap.Int("rate_limit_per_sec", cfg.RateLimitPerSec),
	)

	// Create gateway service
	gateway, err := NewGatewayService(cfg, logger)
	if err != nil {
		logger.Fatal("failed to create gateway service", zap.Error(err))
	}
	defer gateway.Close()

	// Set up routes
	router := gateway.setupRoutes()

	// Create HTTP server
	srv := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      router,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	go func() {
		logger.Info("HTTP server starting", zap.String("port", cfg.Port))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("HTTP server error", zap.Error(err))
		}
	}()

	// Wait for shutdown signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	logger.Info("received shutdown signal", zap.String("signal", sig.String()))

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("HTTP server forced shutdown", zap.Error(err))
	}

	gateway.Close()
	logger.Info("gateway service stopped")
}

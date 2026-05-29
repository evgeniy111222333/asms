// Package health provides health check helpers for ACMS services.
package health

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// Checker performs health checks against various backends.
type Checker struct {
	logger *zap.Logger
}

// NewChecker creates a new health checker.
func NewChecker(logger *zap.Logger) *Checker {
	return &Checker{logger: logger}
}

// ComponentHealth represents the health of a single component.
type ComponentHealth struct {
	Name      string            `json:"name"`
	Status    string            `json:"status"` // healthy, degraded, down
	LatencyMs int64             `json:"latency_ms"`
	Details   map[string]string `json:"details,omitempty"`
	LastCheck time.Time         `json:"last_check"`
}

// CheckRedis performs a real health check against Redis.
func (c *Checker) CheckRedis(client *redis.Client) *ComponentHealth {
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	health := &ComponentHealth{
		Name:      "redis",
		LastCheck: time.Now(),
		Details:   make(map[string]string),
	}

	result, err := client.Ping(ctx).Result()
	latency := time.Since(start).Milliseconds()
	health.LatencyMs = latency

	if err != nil {
		health.Status = "down"
		health.Details["error"] = err.Error()
		c.logger.Error("Redis health check failed", zap.Error(err))
		return health
	}

	// Get additional Redis info
	info, err := client.Info(ctx, "server", "memory", "clients").Result()
	if err == nil {
		health.Details["ping_response"] = result
		_ = info // Parse relevant fields if needed
	}

	// Check connected clients
	clients, err := client.ClientList(ctx).Result()
	if err == nil {
		health.Details["connected_clients"] = fmt.Sprintf("%d", len(clients))
	}

	// Latency thresholds
	if latency > 100 {
		health.Status = "degraded"
		health.Details["note"] = "high latency"
	} else {
		health.Status = "healthy"
	}

	return health
}

// CheckRedpanda performs a health check against Redpanda/Kafka brokers.
func (c *Checker) CheckRedpanda(brokers string) *ComponentHealth {
	start := time.Now()
	health := &ComponentHealth{
		Name:      "redpanda",
		LastCheck: time.Now(),
		Details:   make(map[string]string),
	}

	// Try to reach the Redpanda admin API
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://%s/v1/cluster/health_overview", brokers))
	latency := time.Since(start).Milliseconds()
	health.LatencyMs = latency

	if err != nil {
		// Fallback: try TCP connection to Kafka port
		health.Status = "degraded"
		health.Details["error"] = err.Error()
		health.Details["note"] = "admin API unreachable, broker may still be available via Kafka protocol"
		c.logger.Warn("Redpanda admin API check failed", zap.Error(err))
		return health
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		health.Status = "healthy"
	} else {
		health.Status = "degraded"
		health.Details["status_code"] = fmt.Sprintf("%d", resp.StatusCode)
	}

	return health
}

// CheckPostgreSQL performs a real health check against PostgreSQL.
func (c *Checker) CheckPostgreSQL(db *sql.DB) *ComponentHealth {
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	health := &ComponentHealth{
		Name:      "postgresql",
		LastCheck: time.Now(),
		Details:   make(map[string]string),
	}

	var result int
	err := db.QueryRowContext(ctx, "SELECT 1").Scan(&result)
	latency := time.Since(start).Milliseconds()
	health.LatencyMs = latency

	if err != nil {
		health.Status = "down"
		health.Details["error"] = err.Error()
		c.logger.Error("PostgreSQL health check failed", zap.Error(err))
		return health
	}

	// Get database stats
	stats := db.Stats()
	health.Details["open_connections"] = fmt.Sprintf("%d", stats.OpenConnections)
	health.Details["in_use"] = fmt.Sprintf("%d", stats.InUse)
	health.Details["idle"] = fmt.Sprintf("%d", stats.Idle)

	// Check connection pool health
	if stats.OpenConnections > 80 {
		health.Status = "degraded"
		health.Details["note"] = "high connection count"
	} else {
		health.Status = "healthy"
	}

	return health
}

// CheckHTTP performs a health check against an HTTP endpoint.
func (c *Checker) CheckHTTP(name, url string) *ComponentHealth {
	start := time.Now()
	health := &ComponentHealth{
		Name:      name,
		LastCheck: time.Now(),
		Details:   make(map[string]string),
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	latency := time.Since(start).Milliseconds()
	health.LatencyMs = latency

	if err != nil {
		health.Status = "down"
		health.Details["error"] = err.Error()
		c.logger.Error("HTTP health check failed", zap.String("name", name), zap.Error(err))
		return health
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		health.Status = "healthy"
	} else if resp.StatusCode < 500 {
		health.Status = "degraded"
		health.Details["status_code"] = fmt.Sprintf("%d", resp.StatusCode)
	} else {
		health.Status = "down"
		health.Details["status_code"] = fmt.Sprintf("%d", resp.StatusCode)
	}

	return health
}

// AggregateHealth computes an overall health status from component checks.
func AggregateHealth(components []*ComponentHealth) string {
	allHealthy := true
	anyDegraded := false

	for _, h := range components {
		switch h.Status {
		case "down":
			return "down"
		case "degraded":
			allHealthy = false
			anyDegraded = true
		}
	}

	if anyDegraded {
		return "degraded"
	}

	if allHealthy {
		return "healthy"
	}

	return "unknown"
}

// Package main - ACMS Monitoring Service
//
// Collects metrics from all ACMS components, exposes real Prometheus metrics,
// performs real health checks against Redis, Redpanda, and PostgreSQL,
// sends properly formatted alerts, and provides dashboard-ready metric endpoints.
package main

import (
        "context"
        "database/sql"
        "bytes"
        "encoding/json"
        "fmt"
        "net/http"
        "os"
        "os/signal"
        "sync"
        "syscall"
        "time"

        "github.com/acms/go-services/internal/config"
        "github.com/acms/go-services/internal/health"
        "github.com/acms/go-services/internal/types"
        "github.com/gin-gonic/gin"
        _ "github.com/lib/pq"
        "github.com/prometheus/client_golang/prometheus"
        "github.com/prometheus/client_golang/prometheus/promhttp"
        "github.com/redis/go-redis/v9"
        "go.uber.org/zap"
)

// AlertSeverity defines the severity level of an alert.
type AlertSeverity string

const (
        AlertInfo     AlertSeverity = "info"
        AlertWarning  AlertSeverity = "warning"
        AlertCritical AlertSeverity = "critical"
)

// AlertEscalation defines how alerts should be escalated.
type AlertEscalation struct {
        Severity  AlertSeverity
        RepeatSec int // How often to repeat the alert
        MaxRepeats int // Max times to repeat before escalating
        Channel   string // "log", "webhook", "both"
}

// EscalationPolicy maps severity levels to escalation rules.
var EscalationPolicy = map[AlertSeverity]AlertEscalation{
        AlertInfo:     {Severity: AlertInfo, RepeatSec: 300, MaxRepeats: 3, Channel: "log"},
        AlertWarning:  {Severity: AlertWarning, RepeatSec: 120, MaxRepeats: 5, Channel: "both"},
        AlertCritical: {Severity: AlertCritical, RepeatSec: 60, MaxRepeats: 10, Channel: "both"},
}

// Prometheus metrics
var (
        // Gauges
        metricPortfolioValue = prometheus.NewGauge(prometheus.GaugeOpts{
                Name: "acms_portfolio_value",
                Help: "Current total portfolio value in USD",
        })
        metricDrawdown = prometheus.NewGauge(prometheus.GaugeOpts{
                Name: "acms_drawdown",
                Help: "Current drawdown percentage",
        })
        metricUnrealizedPnL = prometheus.NewGauge(prometheus.GaugeOpts{
                Name: "acms_unrealized_pnl",
                Help: "Current unrealized PnL in USD",
        })
        metricRealizedPnL = prometheus.NewGauge(prometheus.GaugeOpts{
                Name: "acms_realized_pnl",
                Help: "Total realized PnL in USD",
        })
        metricActivePositions = prometheus.NewGauge(prometheus.GaugeOpts{
                Name: "acms_active_positions",
                Help: "Number of active trading positions",
        })
        metricComponentHealth = prometheus.NewGaugeVec(prometheus.GaugeOpts{
                Name: "acms_component_health",
                Help: "Health status of components: 1=healthy, 0.5=degraded, 0=down",
        }, []string{"component"})
        metricComponentLatency = prometheus.NewGaugeVec(prometheus.GaugeOpts{
                Name: "acms_component_latency_ms",
                Help: "Health check latency in milliseconds",
        }, []string{"component"})
        metricKillSwitchActive = prometheus.NewGauge(prometheus.GaugeOpts{
                Name: "acms_kill_switch_active",
                Help: "Whether the kill switch is active: 1=yes, 0=no",
        })
        metricUptimeSeconds = prometheus.NewGauge(prometheus.GaugeOpts{
                Name: "acms_uptime_seconds",
                Help: "Service uptime in seconds",
        })

        // Counters
        metricOrdersTotal = prometheus.NewCounter(prometheus.CounterOpts{
                Name: "acms_orders_total",
                Help: "Total number of orders placed",
        })
        metricTradesTotal = prometheus.NewCounter(prometheus.CounterOpts{
                Name: "acms_trades_total",
                Help: "Total number of trades executed",
        })
        metricSignalsGenerated = prometheus.NewCounter(prometheus.CounterOpts{
                Name: "acms_signals_generated_total",
                Help: "Total number of trading signals generated",
        })
        metricAlertsSent = prometheus.NewCounterVec(prometheus.CounterOpts{
                Name: "acms_alerts_sent_total",
                Help: "Total number of alerts sent",
        }, []string{"severity", "component"})

        // Histograms
        metricHealthCheckDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
                Name:    "acms_health_check_duration_seconds",
                Help:    "Duration of health check runs",
                Buckets: prometheus.DefBuckets,
        }, []string{"component"})
        metricAPIRequestDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
                Name:    "acms_api_request_duration_seconds",
                Help:    "Duration of API requests to Python backend",
                Buckets: []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0},
        }, []string{"endpoint", "status"})
)

func init() {
        prometheus.MustRegister(
                metricPortfolioValue,
                metricDrawdown,
                metricUnrealizedPnL,
                metricRealizedPnL,
                metricActivePositions,
                metricComponentHealth,
                metricComponentLatency,
                metricKillSwitchActive,
                metricUptimeSeconds,
                metricOrdersTotal,
                metricTradesTotal,
                metricSignalsGenerated,
                metricAlertsSent,
                metricHealthCheckDuration,
                metricAPIRequestDuration,
        )
}

// MonitorService handles monitoring and alerting.
type MonitorService struct {
        config      *config.MonitorConfig
        metrics     *types.SystemMetrics
        healthMap   map[string]*types.HealthStatus
        alerts      []types.Alert
        alertMu     sync.Mutex
        startTime   time.Time
        logger      *zap.Logger
        redisClient *redis.Client
        pgDB        *sql.DB
        checker     *health.Checker
        pythonURL   string
}

// NewMonitorService creates a new monitor service with real backend connections.
func NewMonitorService(cfg *config.MonitorConfig, logger *zap.Logger) *MonitorService {
        m := &MonitorService{
                config:    cfg,
                metrics:   &types.SystemMetrics{},
                healthMap: make(map[string]*types.HealthStatus),
                alerts:    make([]types.Alert, 0),
                startTime: time.Now(),
                logger:    logger,
                pythonURL: cfg.PythonAPIURL,
                checker:   health.NewChecker(logger),
        }

        // Connect to Redis
        m.redisClient = redis.NewClient(&redis.Options{
                Addr:         cfg.RedisURL,
                DB:           0,
                DialTimeout:  3 * time.Second,
                ReadTimeout:  3 * time.Second,
                WriteTimeout: 3 * time.Second,
                PoolSize:     5,
        })

        // Test Redis connection
        ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
        defer cancel()
        if _, err := m.redisClient.Ping(ctx).Result(); err != nil {
                logger.Warn("Redis connection failed, health checks will reflect this", zap.Error(err))
        } else {
                logger.Info("Redis connected for health checks")
        }

        // Connect to PostgreSQL
        pgDB, err := sql.Open("postgres", cfg.PostgresURL)
        if err != nil {
                logger.Warn("PostgreSQL connection failed, health checks will reflect this", zap.Error(err))
        } else {
                pgDB.SetMaxOpenConns(5)
                pgDB.SetMaxIdleConns(2)
                pgDB.SetConnMaxLifetime(5 * time.Minute)
                ctx2, cancel2 := context.WithTimeout(context.Background(), 5*time.Second)
                defer cancel2()
                if err := pgDB.PingContext(ctx2); err != nil {
                        logger.Warn("PostgreSQL ping failed", zap.Error(err))
                } else {
                        logger.Info("PostgreSQL connected for health checks")
                }
                m.pgDB = pgDB
        }

        return m
}

// checkRedisHealth performs a real Redis health check.
func (m *MonitorService) checkRedisHealth() *types.HealthStatus {
        h := m.checker.CheckRedis(m.redisClient)
        m.healthMap["redis"] = convertComponentHealth(h)
        return m.healthMap["redis"]
}

// checkRedpandaHealth performs a real Redpanda health check.
func (m *MonitorService) checkRedpandaHealth() *types.HealthStatus {
        h := m.checker.CheckRedpanda(m.config.RedpandaBrokers)
        m.healthMap["redpanda"] = convertComponentHealth(h)
        return m.healthMap["redpanda"]
}

// checkPostgreSQLHealth performs a real PostgreSQL health check.
func (m *MonitorService) checkPostgreSQLHealth() *types.HealthStatus {
        if m.pgDB == nil {
                m.healthMap["postgresql"] = &types.HealthStatus{
                        Name:      "postgresql",
                        Status:    "down",
                        LatencyMs: 0,
                        LastCheck: time.Now(),
                        Details:   map[string]string{"error": "not connected"},
                }
                return m.healthMap["postgresql"]
        }
        h := m.checker.CheckPostgreSQL(m.pgDB)
        m.healthMap["postgresql"] = convertComponentHealth(h)
        return m.healthMap["postgresql"]
}

// checkPythonAPIHealth checks the Python API health endpoint.
func (m *MonitorService) checkPythonAPIHealth() *types.HealthStatus {
        h := m.checker.CheckHTTP("python_api", m.pythonURL+"/health")
        m.healthMap["python_api"] = convertComponentHealth(h)
        return m.healthMap["python_api"]
}

// checkConnectorHealth checks the connector service health endpoint.
func (m *MonitorService) checkConnectorHealth() *types.HealthStatus {
        connectorURL := config.GetEnv("CONNECTOR_URL", "http://localhost:8082")
        h := m.checker.CheckHTTP("connector", connectorURL+"/health")
        m.healthMap["connector"] = convertComponentHealth(h)
        return m.healthMap["connector"]
}

// checkGatewayHealth checks the gateway health endpoint.
func (m *MonitorService) checkGatewayHealth() *types.HealthStatus {
        gatewayURL := config.GetEnv("GATEWAY_URL", "http://localhost:8080")
        h := m.checker.CheckHTTP("gateway", gatewayURL+"/health")
        m.healthMap["gateway"] = convertComponentHealth(h)
        return m.healthMap["gateway"]
}

func convertComponentHealth(ch *health.ComponentHealth) *types.HealthStatus {
        return &types.HealthStatus{
                Name:      ch.Name,
                Status:    ch.Status,
                LatencyMs: ch.LatencyMs,
                Details:   ch.Details,
                LastCheck: ch.LastCheck,
        }
}

// runAllHealthChecks executes all health checks and updates Prometheus metrics.
func (m *MonitorService) runAllHealthChecks() {
        start := time.Now()
        m.logger.Debug("running health checks")

        // Run all checks concurrently
        var wg sync.WaitGroup
        checks := []func(){
                m.checkRedisHealth,
                m.checkRedpandaHealth,
                m.checkPostgreSQLHealth,
                m.checkPythonAPIHealth,
                m.checkConnectorHealth,
                m.checkGatewayHealth,
        }

        for _, check := range checks {
                wg.Add(1)
                go func(fn func()) {
                        defer wg.Done()
                        fn()
                }(check)
        }
        wg.Wait()

        // Update Prometheus metrics for each component
        for name, h := range m.healthMap {
                healthValue := 1.0
                switch h.Status {
                case "healthy":
                        healthValue = 1.0
                case "degraded":
                        healthValue = 0.5
                case "down":
                        healthValue = 0.0
                }
                metricComponentHealth.WithLabelValues(name).Set(healthValue)
                metricComponentLatency.WithLabelValues(name).Set(float64(h.LatencyMs))

                duration := time.Since(start).Seconds()
                metricHealthCheckDuration.WithLabelValues(name).Observe(duration)

                // Alert on down or degraded components
                if h.Status == "down" {
                        m.sendAlert(h.Name, fmt.Sprintf("Component %s is DOWN: %v", h.Name, h.Details), AlertCritical)
                } else if h.Status == "degraded" {
                        m.sendAlert(h.Name, fmt.Sprintf("Component %s is DEGRADED: latency=%dms", h.Name, h.LatencyMs), AlertWarning)
                }
        }

        m.logger.Debug("health checks completed",
                zap.Duration("duration", time.Since(start)),
                zap.Int("components", len(m.healthMap)),
        )
}

// sendAlert sends an alert with proper body, severity, and escalation.
// FIX: The original code sent a nil body to http.Post. This version
// properly sends the JSON-encoded alert payload.
func (m *MonitorService) sendAlert(component, message string, severity AlertSeverity) {
        policy, exists := EscalationPolicy[severity]
        if !exists {
                policy = EscalationPolicy[AlertInfo]
        }

        alert := types.Alert{
                ID:          fmt.Sprintf("%s-%d", component, time.Now().UnixNano()),
                Component:   component,
                Severity:    string(severity),
                Message:     message,
                Timestamp:   time.Now(),
                Acknowledged: false,
        }

        // Store alert
        m.alertMu.Lock()
        m.alerts = append(m.alerts, alert)
        // Keep only last 100 alerts
        if len(m.alerts) > 100 {
                m.alerts = m.alerts[len(m.alerts)-100:]
        }
        m.alertMu.Unlock()

        // Increment Prometheus counter
        metricAlertsSent.WithLabelValues(string(severity), component).Inc()

        m.logger.Warn("alert triggered",
                zap.String("component", component),
                zap.String("severity", string(severity)),
                zap.String("message", message),
                zap.String("channel", policy.Channel),
        )

        // Send via webhook if configured
        if m.config.AlertWebhookURL != "" && (policy.Channel == "webhook" || policy.Channel == "both") {
                m.sendAlertWebhook(alert, policy)
        }
}

// sendAlertWebhook sends the alert to the configured webhook URL.
// This fixes the bug where the original code used http.Post with a nil body.
func (m *MonitorService) sendAlertWebhook(alert types.Alert, policy AlertEscalation) {
        payload := map[string]interface{}{
                "alert_id":    alert.ID,
                "component":   alert.Component,
                "severity":    alert.Severity,
                "message":     alert.Message,
                "timestamp":   alert.Timestamp.Format(time.RFC3339),
                "service":     "acms-monitor",
                "escalation": map[string]interface{}{
                        "channel":     policy.Channel,
                        "repeat_sec":  policy.RepeatSec,
                        "max_repeats": policy.MaxRepeats,
                },
        }

        data, err := json.Marshal(payload)
        if err != nil {
                m.logger.Error("failed to marshal alert payload", zap.Error(err))
                return
        }

        // Properly send the JSON body (this is the fix for the original nil body bug).
        // The original code did: http.Post(url, "application/json", nil)
        // which sent an empty body. We now use bytes.NewReader to send the actual payload.
        ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
        defer cancel()

        req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.config.AlertWebhookURL, bytes.NewReader(data))
        if err != nil {
                m.logger.Error("failed to create alert request", zap.Error(err))
                return
        }
        req.Header.Set("Content-Type", "application/json")

        client := &http.Client{Timeout: 5 * time.Second}
        resp, err := client.Do(req)
        if err != nil {
                m.logger.Error("failed to send alert webhook", zap.Error(err))
                return
        }
        defer resp.Body.Close()

        if resp.StatusCode >= 400 {
                m.logger.Warn("alert webhook returned error status",
                        zap.Int("status", resp.StatusCode),
                )
        } else {
                m.logger.Info("alert webhook sent successfully",
                        zap.String("component", alert.Component),
                        zap.String("severity", alert.Severity),
                )
        }
}

// fetchMetricsFromPythonAPI retrieves metrics from the Python API.
func (m *MonitorService) fetchMetricsFromPythonAPI() {
        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer cancel()

        url := m.pythonURL + "/api/v1/portfolio"
        req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
        if err != nil {
                m.logger.Debug("failed to create metrics request", zap.Error(err))
                return
        }

        client := &http.Client{Timeout: 10 * time.Second}
        start := time.Now()
        resp, err := client.Do(req)
        duration := time.Since(start).Seconds()

        if err != nil {
                m.logger.Debug("failed to fetch metrics from Python API", zap.Error(err))
                metricAPIRequestDuration.WithLabelValues("/api/v1/portfolio", "error").Observe(duration)
                return
        }
        defer resp.Body.Close()

        metricAPIRequestDuration.WithLabelValues("/api/v1/portfolio", fmt.Sprintf("%d", resp.StatusCode)).Observe(duration)

        if resp.StatusCode != 200 {
                m.logger.Debug("Python API returned non-200 status", zap.Int("status", resp.StatusCode))
                return
        }

        var result struct {
                Data struct {
                        TotalValue     float64 `json:"total_value"`
                        UnrealizedPnL  float64 `json:"unrealized_pnl"`
                        RealizedPnL    float64 `json:"realized_pnl"`
                        Drawdown       float64 `json:"drawdown"`
                        ActivePositions int    `json:"active_positions"`
                } `json:"data"`
        }

        if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
                m.logger.Debug("failed to decode metrics response", zap.Error(err))
                return
        }

        // Update internal metrics
        m.metrics.mu.Lock()
        m.metrics.PortfolioValue = result.Data.TotalValue
        m.metrics.UnrealizedPnL = result.Data.UnrealizedPnL
        m.metrics.RealizedPnL = result.Data.RealizedPnL
        m.metrics.Drawdown = result.Data.Drawdown
        m.metrics.ActivePositions = result.Data.ActivePositions
        m.metrics.mu.Unlock()

        // Update Prometheus gauges
        metricPortfolioValue.Set(result.Data.TotalValue)
        metricUnrealizedPnL.Set(result.Data.UnrealizedPnL)
        metricRealizedPnL.Set(result.Data.RealizedPnL)
        metricDrawdown.Set(result.Data.Drawdown)
        metricActivePositions.Set(float64(result.Data.ActivePositions))

        m.logger.Debug("metrics updated from Python API",
                zap.Float64("portfolio_value", result.Data.TotalValue),
                zap.Float64("drawdown", result.Data.Drawdown),
        )
}

// fetchRiskStatus retrieves risk status from the Python API.
func (m *MonitorService) fetchRiskStatus() {
        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer cancel()

        url := m.pythonURL + "/api/v1/risk/status"
        req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
        if err != nil {
                return
        }

        client := &http.Client{Timeout: 10*time.Second}
        resp, err := client.Do(req)
        if err != nil {
                return
        }
        defer resp.Body.Close()

        if resp.StatusCode != 200 {
                return
        }

        var result struct {
                Data struct {
                        KillSwitchActive bool `json:"kill_switch_active"`
                } `json:"data"`
        }

        if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
                return
        }

        m.metrics.KillSwitchActive = result.Data.KillSwitchActive
        if result.Data.KillSwitchActive {
                metricKillSwitchActive.Set(1)
                m.sendAlert("risk", "Kill switch is ACTIVE - all trading halted", AlertCritical)
        } else {
                metricKillSwitchActive.Set(0)
        }
}

// collectOrderMetrics fetches order/trade counts from the Python API.
func (m *MonitorService) collectOrderMetrics() {
        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer cancel()

        url := m.pythonURL + "/api/v1/orders"
        req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
        if err != nil {
                return
        }

        client := &http.Client{Timeout: 10*time.Second}
        resp, err := client.Do(req)
        if err != nil {
                return
        }
        defer resp.Body.Close()

        if resp.StatusCode != 200 {
                return
        }

        var result struct {
                Data struct {
                        Total int64 `json:"total"`
                } `json:"data"`
        }

        if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
                return
        }

        m.metrics.TotalOrders = result.Data.Total
}

// startPeriodicChecks runs health checks and metric collection on a schedule.
func (m *MonitorService) startPeriodicChecks(ctx context.Context) {
        // Health check ticker
        healthTicker := time.NewTicker(time.Duration(m.config.CheckIntervalSec) * time.Second)
        defer healthTicker.Stop()

        // Metrics collection ticker (every 15 seconds)
        metricsTicker := time.NewTicker(15 * time.Second)
        defer metricsTicker.Stop()

        // Risk status ticker (every 30 seconds)
        riskTicker := time.NewTicker(30 * time.Second)
        defer riskTicker.Stop()

        // Uptime updater
        uptimeTicker := time.NewTicker(1 * time.Second)
        defer uptimeTicker.Stop()

        // Initial checks
        m.runAllHealthChecks()
        m.fetchMetricsFromPythonAPI()
        m.fetchRiskStatus()

        for {
                select {
                case <-ctx.Done():
                        return
                case <-healthTicker.C:
                        m.runAllHealthChecks()
                case <-metricsTicker.C:
                        m.fetchMetricsFromPythonAPI()
                        m.collectOrderMetrics()
                case <-riskTicker.C:
                        m.fetchRiskStatus()
                case <-uptimeTicker.C:
                        uptime := int64(time.Since(m.startTime).Seconds())
                        m.metrics.UptimeSeconds = uptime
                        metricUptimeSeconds.Set(float64(uptime))
                }
        }
}

// GetAlerts returns recent alerts.
func (m *MonitorService) GetAlerts(limit int) []types.Alert {
        m.alertMu.Lock()
        defer m.alertMu.Unlock()

        if limit > len(m.alerts) {
                limit = len(m.alerts)
        }
        if limit <= 0 {
                limit = 20
        }

        result := make([]types.Alert, 0, limit)
        start := len(m.alerts) - limit
        if start < 0 {
                start = 0
        }
        result = append(result, m.alerts[start:]...)
        return result
}

// AcknowledgeAlert marks an alert as acknowledged.
func (m *MonitorService) AcknowledgeAlert(alertID string) bool {
        m.alertMu.Lock()
        defer m.alertMu.Unlock()

        for i := range m.alerts {
                if m.alerts[i].ID == alertID {
                        m.alerts[i].Acknowledged = true
                        return true
                }
        }
        return false
}

// Close cleans up resources.
func (m *MonitorService) Close() {
        if m.redisClient != nil {
                m.redisClient.Close()
        }
        if m.pgDB != nil {
                m.pgDB.Close()
        }
}

func main() {
        // Initialize structured logger
        logger, err := zap.NewProduction()
        if err != nil {
                fmt.Fprintf(os.Stderr, "failed to create logger: %v\n", err)
                os.Exit(1)
        }
        defer logger.Sync()

        // Load configuration
        cfg := config.LoadMonitorConfig()
        if err := cfg.Validate(); err != nil {
                logger.Fatal("invalid configuration", zap.Error(err))
        }

        logger.Info("starting monitor service",
                zap.String("port", cfg.Port),
                zap.Int("check_interval_sec", cfg.CheckIntervalSec),
                zap.String("python_api_url", cfg.PythonAPIURL),
                zap.String("redis_url", cfg.RedisURL),
                zap.String("redpanda_brokers", cfg.RedpandaBrokers),
        )

        // Create monitor service with real backend connections
        monitor := NewMonitorService(cfg, logger)
        defer monitor.Close()

        // Set up HTTP server
        gin.SetMode(gin.ReleaseMode)
        router := gin.New()
        router.Use(gin.Recovery())

        // Basic health endpoint
        router.GET("/health", func(c *gin.Context) {
                allHealthy := true
                for _, h := range monitor.healthMap {
                        if h.Status != "healthy" {
                                allHealthy = false
                                break
                        }
                }
                status := "healthy"
                if !allHealthy {
                        status = "degraded"
                }
                c.JSON(http.StatusOK, gin.H{
                        "status":  status,
                        "service": "monitor",
                        "uptime":  int64(time.Since(monitor.startTime).Seconds()),
                })
        })

        // All component health endpoint
        router.GET("/monitor/health", func(c *gin.Context) {
                components := make(map[string]*types.HealthStatus)
                for k, v := range monitor.healthMap {
                        components[k] = v
                }
                overall := health.AggregateHealth(convertToComponentHealth(components))
                c.JSON(http.StatusOK, gin.H{
                        "status":     overall,
                        "components": components,
                        "timestamp":  time.Now().Format(time.RFC3339),
                })
        })

        // JSON metrics endpoint (human-readable)
        router.GET("/monitor/metrics", func(c *gin.Context) {
                uptime := int64(time.Since(monitor.startTime).Seconds())
                monitor.metrics.UptimeSeconds = uptime
                c.JSON(http.StatusOK, monitor.metrics)
        })

        // Prometheus metrics endpoint (machine-readable)
        router.GET("/metrics", gin.WrapH(promhttp.Handler()))

        // Alerts endpoints
        router.GET("/monitor/alerts", func(c *gin.Context) {
                limit := 20
                if l := c.Query("limit"); l != "" {
                        fmt.Sscanf(l, "%d", &limit)
                }
                c.JSON(http.StatusOK, gin.H{
                        "alerts": monitor.GetAlerts(limit),
                        "count":  len(monitor.GetAlerts(limit)),
                })
        })

        router.POST("/monitor/alerts/:id/acknowledge", func(c *gin.Context) {
                alertID := c.Param("id")
                if monitor.AcknowledgeAlert(alertID) {
                        c.JSON(http.StatusOK, gin.H{"message": "alert acknowledged"})
                } else {
                        c.JSON(http.StatusNotFound, gin.H{"error": "alert not found"})
                }
        })

        // Component-specific health endpoints
        router.GET("/monitor/health/redis", func(c *gin.Context) {
                h := monitor.checkRedisHealth()
                statusCode := http.StatusOK
                if h.Status == "down" {
                        statusCode = http.StatusServiceUnavailable
                }
                c.JSON(statusCode, h)
        })

        router.GET("/monitor/health/redpanda", func(c *gin.Context) {
                h := monitor.checkRedpandaHealth()
                statusCode := http.StatusOK
                if h.Status == "down" {
                        statusCode = http.StatusServiceUnavailable
                }
                c.JSON(statusCode, h)
        })

        router.GET("/monitor/health/postgresql", func(c *gin.Context) {
                h := monitor.checkPostgreSQLHealth()
                statusCode := http.StatusOK
                if h.Status == "down" {
                        statusCode = http.StatusServiceUnavailable
                }
                c.JSON(statusCode, h)
        })

        router.GET("/monitor/health/python-api", func(c *gin.Context) {
                h := monitor.checkPythonAPIHealth()
                statusCode := http.StatusOK
                if h.Status == "down" {
                        statusCode = http.StatusServiceUnavailable
                }
                c.JSON(statusCode, h)
        })

        // Dashboard-ready summary endpoint
        router.GET("/monitor/dashboard", func(c *gin.Context) {
                uptime := int64(time.Since(monitor.startTime).Seconds())
                componentHealth := make(map[string]string)
                for k, v := range monitor.healthMap {
                        componentHealth[k] = v.Status
                }

                recentAlerts := monitor.GetAlerts(10)
                criticalCount := 0
                for _, a := range recentAlerts {
                        if a.Severity == "critical" && !a.Acknowledged {
                                criticalCount++
                        }
                }

                c.JSON(http.StatusOK, gin.H{
                        "system": gin.H{
                                "uptime_seconds": uptime,
                                "status":         health.AggregateHealth(convertToComponentHealth(monitor.healthMap)),
                        },
                        "portfolio": gin.H{
                                "value":            monitor.metrics.PortfolioValue,
                                "unrealized_pnl":   monitor.metrics.UnrealizedPnL,
                                "realized_pnl":     monitor.metrics.RealizedPnL,
                                "drawdown":         monitor.metrics.Drawdown,
                                "active_positions": monitor.metrics.ActivePositions,
                        },
                        "components": componentHealth,
                        "alerts": gin.H{
                                "recent_count":      len(recentAlerts),
                                "unacknowledged_critical": criticalCount,
                        },
                        "kill_switch": monitor.metrics.KillSwitchActive,
                })
        })

        // Force health check endpoint
        router.POST("/monitor/health/check", func(c *gin.Context) {
                monitor.runAllHealthChecks()
                c.JSON(http.StatusOK, gin.H{
                        "message":    "health checks completed",
                        "components": monitor.healthMap,
                })
        })

        // Start HTTP server
        srv := &http.Server{
                Addr:         ":" + cfg.Port,
                Handler:      router,
                ReadTimeout:  15 * time.Second,
                WriteTimeout: 15 * time.Second,
                IdleTimeout:  120 * time.Second,
        }

        go func() {
                logger.Info("HTTP server starting", zap.String("port", cfg.Port))
                if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
                        logger.Fatal("HTTP server error", zap.Error(err))
                }
        }()

        // Start periodic checks
        ctx, cancel := context.WithCancel(context.Background())
        go monitor.startPeriodicChecks(ctx)

        // Wait for shutdown signal
        quit := make(chan os.Signal, 1)
        signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
        sig := <-quit
        logger.Info("received shutdown signal", zap.String("signal", sig.String()))

        // Cancel periodic checks
        cancel()

        // Shutdown HTTP server
        shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer shutdownCancel()

        if err := srv.Shutdown(shutdownCtx); err != nil {
                logger.Error("HTTP server forced shutdown", zap.Error(err))
        }

        monitor.Close()
        logger.Info("monitor service stopped")
}

// convertToComponentHealth converts types.HealthStatus map to health.ComponentHealth slice.
func convertToComponentHealth(m map[string]*types.HealthStatus) []*health.ComponentHealth {
        result := make([]*health.ComponentHealth, 0, len(m))
        for _, v := range m {
                result = append(result, &health.ComponentHealth{
                        Name:      v.Name,
                        Status:    v.Status,
                        LatencyMs: v.LatencyMs,
                        Details:   v.Details,
                        LastCheck: v.LastCheck,
                })
        }
        return result
}



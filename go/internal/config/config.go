// Package config provides shared configuration loading for ACMS Go services.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// GetEnv returns the value of an environment variable or a fallback.
func GetEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// GetEnvInt returns the integer value of an environment variable or a fallback.
func GetEnvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return i
}

// GetEnvDuration returns a duration from an environment variable or fallback.
func GetEnvDuration(key string, fallback time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return fallback
	}
	return d
}

// GetEnvBool returns a boolean from an environment variable or fallback.
func GetEnvBool(key string, fallback bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return fallback
	}
	return b
}

// GatewayConfig holds the API gateway configuration.
type GatewayConfig struct {
	Port              string
	PythonAPIURL      string
	MonitorURL        string
	ConnectorURL      string
	JWTSecret         string
	RateLimitPerSec   int
	RedisURL          string
	CORSHosts         []string
	MaxRequestSize    int64
	MaxResponseSize   int64
	IPWhitelist       []string
	IPBlacklist       []string
	ReadTimeout       time.Duration
	WriteTimeout      time.Duration
	IdleTimeout       time.Duration
}

// LoadGatewayConfig loads gateway configuration from environment.
func LoadGatewayConfig() *GatewayConfig {
	return &GatewayConfig{
		Port:            GetEnv("GATEWAY_PORT", "8080"),
		PythonAPIURL:    GetEnv("PYTHON_API_URL", "http://localhost:8000"),
		MonitorURL:      GetEnv("MONITOR_URL", "http://localhost:8081"),
		ConnectorURL:    GetEnv("CONNECTOR_URL", "http://localhost:8082"),
		JWTSecret:       GetEnv("JWT_SECRET", "change-me-in-production"),
		RateLimitPerSec: GetEnvInt("RATE_LIMIT_PER_SEC", 100),
		RedisURL:        GetEnv("REDIS_URL", "redis://localhost:6379/0"),
		MaxRequestSize:  int64(GetEnvInt("MAX_REQUEST_SIZE", 1<<20)),  // 1MB
		MaxResponseSize: int64(GetEnvInt("MAX_RESPONSE_SIZE", 5<<20)), // 5MB
		ReadTimeout:     GetEnvDuration("GATEWAY_READ_TIMEOUT", 15*time.Second),
		WriteTimeout:    GetEnvDuration("GATEWAY_WRITE_TIMEOUT", 30*time.Second),
		IdleTimeout:     GetEnvDuration("GATEWAY_IDLE_TIMEOUT", 120*time.Second),
	}
}

// MonitorConfig holds the monitor service configuration.
type MonitorConfig struct {
	Port             string
	CheckIntervalSec int
	PythonAPIURL     string
	AlertWebhookURL  string
	RedisURL         string
	RedpandaBrokers  string
	PostgresURL      string
	LogLevel         string
}

// LoadMonitorConfig loads monitor configuration from environment.
func LoadMonitorConfig() *MonitorConfig {
	return &MonitorConfig{
		Port:             GetEnv("MONITOR_PORT", "8081"),
		CheckIntervalSec: GetEnvInt("CHECK_INTERVAL_SEC", 30),
		PythonAPIURL:     GetEnv("PYTHON_API_URL", "http://localhost:8000"),
		AlertWebhookURL:  GetEnv("ALERT_WEBHOOK_URL", ""),
		RedisURL:         GetEnv("REDIS_URL", "redis://localhost:6379/0"),
		RedpandaBrokers:  GetEnv("REDPANDA_BROKERS", "localhost:9092"),
		PostgresURL:      GetEnv("DATABASE_URL", "postgresql://acms:acms@localhost:5432/acms?sslmode=disable"),
		LogLevel:         GetEnv("LOG_LEVEL", "info"),
	}
}

// ConnectorConfig holds the connector service configuration.
type ConnectorConfig struct {
	Port            string
	RedpandaBrokers string
	Symbols         []string
	Exchanges       []string
	ReconnectBaseMs int
	ReconnectMaxMs  int
	LogLevel        string
}

// LoadConnectorConfig loads connector configuration from environment.
func LoadConnectorConfig() *ConnectorConfig {
	return &ConnectorConfig{
		Port:            GetEnv("CONNECTOR_PORT", "8082"),
		RedpandaBrokers: GetEnv("REDPANDA_BROKERS", "localhost:9092"),
		Symbols:         []string{"BTCUSDT", "ETHUSDT", "SOLUSDT"},
		Exchanges:       []string{"binance", "bybit", "okx"},
		ReconnectBaseMs: GetEnvInt("RECONNECT_BASE_MS", 500),
		ReconnectMaxMs:  GetEnvInt("RECONNECT_MAX_MS", 30000),
		LogLevel:        GetEnv("LOG_LEVEL", "info"),
	}
}

// Validate checks that critical config values are present.
func (c *GatewayConfig) Validate() error {
	if c.JWTSecret == "change-me-in-production" {
		fmt.Println("[WARN] Using default JWT secret - change in production!")
	}
	return nil
}

// Validate checks that critical config values are present.
func (c *MonitorConfig) Validate() error {
	if c.RedisURL == "" {
		return fmt.Errorf("REDIS_URL is required")
	}
	return nil
}

// Validate checks that critical config values are present.
func (c *ConnectorConfig) Validate() error {
	if c.RedpandaBrokers == "" {
		return fmt.Errorf("REDPANDA_BROKERS is required")
	}
	return nil
}

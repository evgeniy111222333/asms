// Package types defines shared types used across ACMS Go services.
package types

import (
        "sync"
        "time"
)

// MarketData represents a normalized market data event from any exchange.
type MarketData struct {
        Exchange  string  `json:"exchange"`
        Symbol    string  `json:"symbol"`
        Price     float64 `json:"price"`
        Quantity  float64 `json:"quantity"`
        Side      string  `json:"side"`       // "buy" or "sell"
        Timestamp int64   `json:"timestamp"`  // Unix milliseconds
        TradeID   string  `json:"trade_id"`
        StreamType string `json:"stream_type"` // "trade", "orderbook", "candle"
}

// OrderBookLevel represents a single price level in an order book.
type OrderBookLevel struct {
        Price    float64 `json:"price"`
        Quantity float64 `json:"quantity"`
}

// OrderBookSnapshot represents a full or partial order book snapshot.
type OrderBookSnapshot struct {
        Exchange  string            `json:"exchange"`
        Symbol    string            `json:"symbol"`
        Bids      []OrderBookLevel  `json:"bids"`
        Asks      []OrderBookLevel  `json:"asks"`
        Timestamp int64             `json:"timestamp"`
}

// Candle represents an OHLCV candlestick data point.
type Candle struct {
        Exchange  string  `json:"exchange"`
        Symbol    string  `json:"symbol"`
        Interval  string  `json:"interval"`
        OpenTime  int64   `json:"open_time"`
        Open      float64 `json:"open"`
        High      float64 `json:"high"`
        Low       float64 `json:"low"`
        Close     float64 `json:"close"`
        Volume    float64 `json:"volume"`
        CloseTime int64   `json:"close_time"`
}

// HealthStatus represents the health of a component.
type HealthStatus struct {
        Name      string            `json:"name"`
        Status    string            `json:"status"` // healthy, degraded, down
        LatencyMs int64             `json:"latency_ms"`
        Details   map[string]string `json:"details,omitempty"`
        LastCheck time.Time         `json:"last_check"`
}

// SystemMetrics tracks system-wide metrics.
type SystemMetrics struct {
        mu               sync.RWMutex `json:"-"`
        TotalOrders      int64        `json:"total_orders"`
        TotalTrades      int64        `json:"total_trades"`
        ActivePositions  int          `json:"active_positions"`
        PortfolioValue   float64      `json:"portfolio_value"`
        UnrealizedPnL    float64      `json:"unrealized_pnl"`
        RealizedPnL      float64      `json:"realized_pnl"`
        Drawdown         float64      `json:"drawdown"`
        SignalsGenerated int64        `json:"signals_generated"`
        KillSwitchActive bool         `json:"kill_switch_active"`
        UptimeSeconds    int64        `json:"uptime_seconds"`
}

// Alert represents a monitoring alert.
type Alert struct {
        ID        string    `json:"id"`
        Component string    `json:"component"`
        Severity  string    `json:"severity"` // info, warning, critical
        Message   string    `json:"message"`
        Timestamp time.Time `json:"timestamp"`
        Acknowledged bool   `json:"acknowledged"`
}

// ExchangeConnection holds the state of a connection to an exchange.
type ExchangeConnection struct {
        Name       string    `json:"name"`
        Status     string    `json:"status"` // connected, disconnected, reconnecting
        LastMsg    time.Time `json:"last_msg"`
        MsgCount   int64     `json:"msg_count"`
        Reconnects int       `json:"reconnects"`
        Streams    []string  `json:"streams"`
}

// CircuitState represents the state of a circuit breaker.
type CircuitState int

const (
        CircuitClosed   CircuitState = iota // Normal operation
        CircuitOpen                         // Failing, reject requests
        CircuitHalfOpen                     // Testing recovery
)

func (s CircuitState) String() string {
        switch s {
        case CircuitClosed:
                return "closed"
        case CircuitOpen:
                return "open"
        case CircuitHalfOpen:
                return "half-open"
        default:
                return "unknown"
        }
}

// CircuitBreakerConfig holds circuit breaker configuration.
type CircuitBreakerConfig struct {
        FailureThreshold int           `json:"failure_threshold"`
        ResetTimeout     time.Duration `json:"reset_timeout"`
        HalfOpenMax      int           `json:"half_open_max"`
}

// ServiceEndpoint describes a backend service for routing.
type ServiceEndpoint struct {
        Name    string `json:"name"`
        URL     string `json:"url"`
        Healthy bool   `json:"healthy"`
}

// Package main - ACMS Exchange Connector Service
//
// Manages real WebSocket connections to Binance, Bybit, and OKX exchanges,
// normalizes market data, and publishes to Redpanda/Kafka topics.
// Features auto-reconnect with exponential backoff, multiple market data
// streams (trades, orderbook, candles), and graceful shutdown.
package main

import (
        "context"
        "encoding/json"
        "fmt"
        "net/http"
        "os"
        "os/signal"
        "strconv"
        "sync"
        "sync/atomic"
        "syscall"
        "time"

        "github.com/acms/go-services/internal/config"
        "github.com/acms/go-services/internal/kafka"
        "github.com/acms/go-services/internal/types"
        "github.com/gin-gonic/gin"
        "github.com/gorilla/websocket"
        "go.uber.org/zap"
)

// StreamType identifies the kind of market data stream.
type StreamType string

const (
        StreamTrade     StreamType = "trade"
        StreamOrderBook StreamType = "orderbook"
        StreamCandle    StreamType = "candle"
)

// Kafka topic names for different data streams.
const (
        TopicTrades     = "acms.market.trades"
        TopicOrderBook  = "acms.market.orderbook"
        TopicCandles    = "acms.market.candles"
        TopicHeartbeat  = "acms.market.heartbeat"
)

// ExchangeWSConfig holds the WebSocket URL and message format for an exchange.
type ExchangeWSConfig struct {
        Name       string
        WSURL      string
        StreamType StreamType
}

// ExchangeWSClients maps exchange names to their WebSocket configurations.
var ExchangeWSClients = map[string][]ExchangeWSConfig{
        "binance": {
                {Name: "binance", WSURL: "wss://stream.binance.com:9443/ws", StreamType: StreamTrade},
                {Name: "binance", WSURL: "wss://stream.binance.com:9443/ws", StreamType: StreamOrderBook},
                {Name: "binance", WSURL: "wss://stream.binance.com:9443/ws", StreamType: StreamCandle},
        },
        "bybit": {
                {Name: "bybit", WSURL: "wss://stream.bybit.com/v5/public/spot", StreamType: StreamTrade},
                {Name: "bybit", WSURL: "wss://stream.bybit.com/v5/public/spot", StreamType: StreamOrderBook},
        },
        "okx": {
                {Name: "okx", WSURL: "wss://ws.okx.com:8443/ws/v5/public", StreamType: StreamTrade},
                {Name: "okx", WSURL: "wss://ws.okx.com:8443/ws/v5/public", StreamType: StreamOrderBook},
        },
}

// WSConnection wraps a single WebSocket connection with reconnection logic.
type WSConnection struct {
        mu             sync.Mutex
        exchange       string
        wsURL          string
        streamType     StreamType
        conn           *websocket.Conn
        status         string // connected, disconnected, reconnecting
        msgCount       int64
        lastMsg        time.Time
        reconnects     int
        baseBackoffMs  int
        maxBackoffMs   int
        producer       *kafka.Producer
        logger         *zap.Logger
        stopCh         chan struct{}
        dataChan       chan types.MarketData
        symbols        []string
        connected      atomic.Bool
}

// NewWSConnection creates a new WebSocket connection handler.
func NewWSConnection(exchange, wsURL string, streamType StreamType,
        producer *kafka.Producer, logger *zap.Logger, symbols []string) *WSConnection {
        return &WSConnection{
                exchange:      exchange,
                wsURL:         wsURL,
                streamType:    streamType,
                status:        "disconnected",
                baseBackoffMs: 500,
                maxBackoffMs:  30000,
                producer:      producer,
                logger:        logger,
                stopCh:        make(chan struct{}),
                dataChan:      make(chan types.MarketData, 10000),
                symbols:       symbols,
        }
}

// Connect establishes the WebSocket connection with subscription messages.
func (w *WSConnection) Connect() error {
        w.mu.Lock()
        defer w.mu.Unlock()

        w.status = "reconnecting"
        w.logger.Info("connecting to exchange WebSocket",
                zap.String("exchange", w.exchange),
                zap.String("stream_type", string(w.streamType)),
                zap.String("url", w.wsURL),
        )

        dialer := websocket.Dialer{
                HandshakeTimeout: 10 * time.Second,
        }

        conn, _, err := dialer.Dial(w.wsURL, nil)
        if err != nil {
                w.status = "disconnected"
                return fmt.Errorf("failed to connect to %s WebSocket: %w", w.exchange, err)
        }

        w.conn = conn
        w.status = "connected"
        w.connected.Store(true)
        w.reconnects++
        w.logger.Info("connected to exchange WebSocket",
                zap.String("exchange", w.exchange),
                zap.String("stream_type", string(w.streamType)),
                zap.Int("reconnects", w.reconnects),
        )

        // Send subscription messages based on exchange and stream type
        if err := w.subscribe(); err != nil {
                w.logger.Warn("subscription failed", zap.Error(err))
        }

        return nil
}

// subscribe sends the appropriate subscription message for the exchange.
func (w *WSConnection) subscribe() error {
        switch w.exchange {
        case "binance":
                return w.subscribeBinance()
        case "bybit":
                return w.subscribeBybit()
        case "okx":
                return w.subscribeOKX()
        default:
                return fmt.Errorf("unknown exchange: %s", w.exchange)
        }
}

func (w *WSConnection) subscribeBinance() error {
        switch w.streamType {
        case StreamTrade:
                for _, symbol := range w.symbols {
                        msg := map[string]interface{}{
                                "method": "SUBSCRIBE",
                                "params": []string{fmt.Sprintf("%s@trade", strtolower(symbol))},
                                "id":     time.Now().UnixNano(),
                        }
                        if err := w.conn.WriteJSON(msg); err != nil {
                                return err
                        }
                }
        case StreamOrderBook:
                for _, symbol := range w.symbols {
                        msg := map[string]interface{}{
                                "method": "SUBSCRIBE",
                                "params": []string{fmt.Sprintf("%s@depth20@100ms", strtolower(symbol))},
                                "id":     time.Now().UnixNano(),
                        }
                        if err := w.conn.WriteJSON(msg); err != nil {
                                return err
                        }
                }
        case StreamCandle:
                for _, symbol := range w.symbols {
                        msg := map[string]interface{}{
                                "method": "SUBSCRIBE",
                                "params": []string{fmt.Sprintf("%s@kline_1m", strtolower(symbol))},
                                "id":     time.Now().UnixNano(),
                        }
                        if err := w.conn.WriteJSON(msg); err != nil {
                                return err
                        }
                }
        }
        return nil
}

func (w *WSConnection) subscribeBybit() error {
        args := make([]map[string]string, 0)
        for _, symbol := range w.symbols {
                switch w.streamType {
                case StreamTrade:
                        args = append(args, map[string]string{
                                "symbol": symbol,
                                "topic":  "publicTrade",
                        })
                case StreamOrderBook:
                        args = append(args, map[string]string{
                                "symbol": symbol,
                                "topic":  "orderbook.25",
                        })
                }
        }
        if len(args) > 0 {
                msg := map[string]interface{}{
                        "op":   "subscribe",
                        "args": args,
                }
                return w.conn.WriteJSON(msg)
        }
        return nil
}

func (w *WSConnection) subscribeOKX() error {
        args := make([]map[string]string, 0)
        for _, symbol := range w.symbols {
                instID := symbolToOKXInstID(symbol)
                switch w.streamType {
                case StreamTrade:
                        args = append(args, map[string]string{
                                "channel": "trades",
                                "instId":  instID,
                        })
                case StreamOrderBook:
                        args = append(args, map[string]string{
                                "channel": "books5",
                                "instId":  instID,
                        })
                }
        }
        if len(args) > 0 {
                msg := map[string]interface{}{
                        "op":   "subscribe",
                        "args": args,
                }
                return w.conn.WriteJSON(msg)
        }
        return nil
}

// ReadLoop continuously reads messages from the WebSocket connection.
func (w *WSConnection) ReadLoop(ctx context.Context) {
        defer w.Disconnect()

        for {
                select {
                case <-ctx.Done():
                        return
                case <-w.stopCh:
                        return
                default:
                }

                w.mu.Lock()
                conn := w.conn
                w.mu.Unlock()

                if conn == nil {
                        time.Sleep(100 * time.Millisecond)
                        continue
                }

                _, message, err := conn.ReadMessage()
                if err != nil {
                        if !w.connected.Load() {
                                return
                        }
                        w.logger.Warn("WebSocket read error",
                                zap.String("exchange", w.exchange),
                                zap.Error(err),
                        )
                        w.connected.Store(false)
                        w.status = "disconnected"
                        return
                }

                w.msgCount++
                w.lastMsg = time.Now()

                // Parse and normalize the message
                data := w.parseMessage(message)
                if data != nil {
                        // Publish to Kafka
                        key := []byte(fmt.Sprintf("%s:%s", data.Exchange, data.Symbol))
                        if err := w.producer.Publish(string(w.kafkaTopic()), key, data); err != nil {
                                w.logger.Error("failed to publish to Kafka",
                                        zap.Error(err),
                                        zap.String("exchange", w.exchange),
                                )
                        }
                }
        }
}

// parseMessage converts a raw WebSocket message into normalized MarketData.
func (w *WSConnection) parseMessage(raw []byte) *types.MarketData {
        switch w.exchange {
        case "binance":
                return w.parseBinanceMessage(raw)
        case "bybit":
                return w.parseBybitMessage(raw)
        case "okx":
                return w.parseOKXMessage(raw)
        default:
                return nil
        }
}

func (w *WSConnection) parseBinanceMessage(raw []byte) *types.MarketData {
        switch w.streamType {
        case StreamTrade:
                var msg struct {
                        EventType string `json:"e"`
                        Symbol    string `json:"s"`
                        TradeID   int64  `json:"t"`
                        Price     string `json:"p"`
                        Quantity  string `json:"q"`
                        BuyerOrd  int64  `json:"b"`
                        SellerOrd int64  `json:"a"`
                        Timestamp int64  `json:"T"`
                }
                if err := json.Unmarshal(raw, &msg); err != nil {
                        return nil
                }
                if msg.EventType != "trade" {
                        return nil
                }
                price, _ := strconv.ParseFloat(msg.Price, 64)
                qty, _ := strconv.ParseFloat(msg.Quantity, 64)
                side := "buy"
                if msg.SellerOrd > msg.BuyerOrd {
                        side = "sell"
                }
                return &types.MarketData{
                        Exchange:   "binance",
                        Symbol:     msg.Symbol,
                        Price:      price,
                        Quantity:   qty,
                        Side:       side,
                        Timestamp:  msg.Timestamp,
                        TradeID:    fmt.Sprintf("%d", msg.TradeID),
                        StreamType: "trade",
                }

        case StreamOrderBook:
                var msg struct {
                        EventType string      `json:"e"`
                        Symbol    string      `json:"s"`
                        Bids      [][]string  `json:"bids"`
                        Asks      [][]string  `json:"asks"`
                }
                if err := json.Unmarshal(raw, &msg); err != nil {
                        return nil
                }
                if msg.EventType != "depthUpdate" && msg.Bids == nil {
                        return nil
                }
                bestBidPrice := 0.0
                bestBidQty := 0.0
                if len(msg.Bids) > 0 {
                        bestBidPrice, _ = strconv.ParseFloat(msg.Bids[0][0], 64)
                        bestBidQty, _ = strconv.ParseFloat(msg.Bids[0][1], 64)
                }
                return &types.MarketData{
                        Exchange:   "binance",
                        Symbol:     msg.Symbol,
                        Price:      bestBidPrice,
                        Quantity:   bestBidQty,
                        Side:       "bid",
                        Timestamp:  time.Now().UnixMilli(),
                        StreamType: "orderbook",
                }

        case StreamCandle:
                var msg struct {
                        EventType string `json:"e"`
                        Symbol    string `json:"s"`
                        Kline     struct {
                                StartTime  int64  `json:"t"`
                                CloseTime  int64  `json:"T"`
                                Interval   string `json:"i"`
                                Open       string `json:"o"`
                                High       string `json:"h"`
                                Low        string `json:"l"`
                                Close      string `json:"c"`
                                Volume     string `json:"v"`
                                IsClosed   bool   `json:"x"`
                        } `json:"k"`
                }
                if err := json.Unmarshal(raw, &msg); err != nil {
                        return nil
                }
                if msg.EventType != "kline" {
                        return nil
                }
                close, _ := strconv.ParseFloat(msg.Kline.Close, 64)
                vol, _ := strconv.ParseFloat(msg.Kline.Volume, 64)
                return &types.MarketData{
                        Exchange:   "binance",
                        Symbol:     msg.Symbol,
                        Price:      close,
                        Quantity:   vol,
                        Timestamp:  msg.Kline.StartTime,
                        StreamType: "candle",
                }
        }
        return nil
}

func (w *WSConnection) parseBybitMessage(raw []byte) *types.MarketData {
        var envelope struct {
                Topic string          `json:"topic"`
                Data json.RawMessage `json:"data"`
                Type string          `json:"type"`
        }
        if err := json.Unmarshal(raw, &envelope); err != nil {
                return nil
        }

        switch {
        case envelope.Topic == "publicTrade" || w.streamType == StreamTrade:
                var tradeData struct {
                        Symbol   string `json:"s"`
                        Price    string `json:"p"`
                        Quantity string `json:"v"`
                        Side     string `json:"S"`
                        TradeID  string `json:"i"`
                        Time     int64  `json:"T"`
                }
                if err := json.Unmarshal(envelope.Data, &tradeData); err != nil {
                        return nil
                }
                price, _ := strconv.ParseFloat(tradeData.Price, 64)
                qty, _ := strconv.ParseFloat(tradeData.Quantity, 64)
                return &types.MarketData{
                        Exchange:   "bybit",
                        Symbol:     tradeData.Symbol,
                        Price:      price,
                        Quantity:   qty,
                        Side:       tradeData.Side,
                        Timestamp:  tradeData.Time,
                        TradeID:    tradeData.TradeID,
                        StreamType: "trade",
                }

        case w.streamType == StreamOrderBook:
                var obData struct {
                        Symbol string     `json:"s"`
                        Bids   [][]string `json:"b"`
                        Asks   [][]string `json:"a"`
                }
                if err := json.Unmarshal(envelope.Data, &obData); err != nil {
                        return nil
                }
                bestBid := 0.0
                if len(obData.Bids) > 0 && len(obData.Bids[0]) >= 2 {
                        bestBid, _ = strconv.ParseFloat(obData.Bids[0][0], 64)
                }
                return &types.MarketData{
                        Exchange:   "bybit",
                        Symbol:     obData.Symbol,
                        Price:      bestBid,
                        Timestamp:  time.Now().UnixMilli(),
                        StreamType: "orderbook",
                }
        }
        return nil
}

func (w *WSConnection) parseOKXMessage(raw []byte) *types.MarketData {
        var envelope struct {
                Arg   map[string]string `json:"arg"`
                Data  json.RawMessage   `json:"data"`
                Event string            `json:"event"`
        }
        if err := json.Unmarshal(raw, &envelope); err != nil {
                return nil
        }

        channel := envelope.Arg["channel"]
        instID := envelope.Arg["instId"]

        switch {
        case channel == "trades" || w.streamType == StreamTrade:
                var trades []struct {
                        TradeID  string `json:"tradeId"`
                        Price    string `json:"px"`
                        Quantity string `json:"sz"`
                        Side     string `json:"side"`
                        Time     string `json:"ts"`
                }
                if err := json.Unmarshal(envelope.Data, &trades); err != nil || len(trades) == 0 {
                        return nil
                }
                t := trades[0]
                price, _ := strconv.ParseFloat(t.Price, 64)
                qty, _ := strconv.ParseFloat(t.Quantity, 64)
                ts, _ := strconv.ParseInt(t.Time, 10, 64)
                return &types.MarketData{
                        Exchange:   "okx",
                        Symbol:     instID,
                        Price:      price,
                        Quantity:   qty,
                        Side:       t.Side,
                        Timestamp:  ts,
                        TradeID:    t.TradeID,
                        StreamType: "trade",
                }

        case channel == "books5" || w.streamType == StreamOrderBook:
                var books []struct {
                        Asks [][]string `json:"asks"`
                        Bids [][]string `json:"bids"`
                        Ts   string     `json:"ts"`
                }
                if err := json.Unmarshal(envelope.Data, &books); err != nil || len(books) == 0 {
                        return nil
                }
                b := books[0]
                bestBid := 0.0
                if len(b.Bids) > 0 && len(b.Bids[0]) >= 2 {
                        bestBid, _ = strconv.ParseFloat(b.Bids[0][0], 64)
                }
                ts, _ := strconv.ParseInt(b.Ts, 10, 64)
                return &types.MarketData{
                        Exchange:   "okx",
                        Symbol:     instID,
                        Price:      bestBid,
                        Timestamp:  ts,
                        StreamType: "orderbook",
                }
        }
        return nil
}

func (w *WSConnection) kafkaTopic() StreamType {
        switch w.streamType {
        case StreamTrade:
                return StreamType(TopicTrades)
        case StreamOrderBook:
                return StreamType(TopicOrderBook)
        case StreamCandle:
                return StreamType(TopicCandles)
        default:
                return StreamType(TopicTrades)
        }
}

// Disconnect cleanly closes the WebSocket connection.
func (w *WSConnection) Disconnect() {
        w.mu.Lock()
        defer w.mu.Unlock()

        w.connected.Store(false)
        w.status = "disconnected"

        if w.conn != nil {
                w.conn.WriteMessage(websocket.CloseMessage,
                        websocket.FormatCloseMessage(websocket.CloseNormalClosure, "shutting down"))
                w.conn.Close()
                w.conn = nil
        }

        w.logger.Info("disconnected from exchange WebSocket",
                zap.String("exchange", w.exchange),
                zap.String("stream_type", string(w.streamType)),
        )
}

// GetStatus returns the current connection status.
func (w *WSConnection) GetStatus() *types.ExchangeConnection {
        w.mu.Lock()
        defer w.mu.Unlock()

        streams := []string{string(w.streamType)}
        return &types.ExchangeConnection{
                Name:       w.exchange,
                Status:     w.status,
                LastMsg:    w.lastMsg,
                MsgCount:   w.msgCount,
                Reconnects: w.reconnects,
                Streams:    streams,
        }
}

// ConnectorService manages all exchange WebSocket connections.
type ConnectorService struct {
        mu         sync.RWMutex
        config     *config.ConnectorConfig
        producer   *kafka.Producer
        logger     *zap.Logger
        connections map[string]*WSConnection
        ctx        context.Context
        cancel     context.CancelFunc
}

// NewConnectorService creates a new connector service.
func NewConnectorService(cfg *config.ConnectorConfig, logger *zap.Logger) (*ConnectorService, error) {
        // Create Kafka producer
        producer, err := kafka.NewProducer(cfg.RedpandaBrokers, logger)
        if err != nil {
                logger.Warn("failed to create Kafka producer, running in log-only mode", zap.Error(err))
        }

        // Ensure topics exist
        if producer != nil {
                topics := []string{TopicTrades, TopicOrderBook, TopicCandles, TopicHeartbeat}
                kafka.EnsureTopics(cfg.RedpandaBrokers, topics, logger)
        }

        ctx, cancel := context.WithCancel(context.Background())

        return &ConnectorService{
                config:      cfg,
                producer:    producer,
                logger:      logger,
                connections: make(map[string]*WSConnection),
                ctx:         ctx,
                cancel:      cancel,
        }, nil
}

// ConnectAll establishes connections to all configured exchanges.
func (s *ConnectorService) ConnectAll() {
        for _, exchange := range s.config.Exchanges {
                s.ConnectExchange(exchange)
        }
}

// ConnectExchange establishes all stream connections for an exchange.
func (s *ConnectorService) ConnectExchange(exchange string) {
        streams, ok := ExchangeWSClients[exchange]
        if !ok {
                s.logger.Error("unknown exchange", zap.String("exchange", exchange))
                return
        }

        for _, streamCfg := range streams {
                key := fmt.Sprintf("%s:%s", exchange, streamCfg.StreamType)
                conn := NewWSConnection(
                        streamCfg.Name,
                        streamCfg.WSURL,
                        streamCfg.StreamType,
                        s.producer,
                        s.logger,
                        s.config.Symbols,
                )

                s.mu.Lock()
                s.connections[key] = conn
                s.mu.Unlock()

                // Start connection loop with auto-reconnect
                go s.connectionLoop(conn)
        }
}

// connectionLoop manages a single connection with exponential backoff reconnect.
func (s *ConnectorService) connectionLoop(conn *WSConnection) {
        backoffMs := conn.baseBackoffMs

        for {
                select {
                case <-s.ctx.Done():
                        conn.Disconnect()
                        return
                default:
                }

                // Attempt connection
                if err := conn.Connect(); err != nil {
                        s.logger.Error("connection failed, backing off",
                                zap.String("exchange", conn.exchange),
                                zap.Error(err),
                                zap.Int("backoff_ms", backoffMs),
                        )

                        select {
                        case <-s.ctx.Done():
                                return
                        case <-time.After(time.Duration(backoffMs) * time.Millisecond):
                        }

                        // Exponential backoff with cap
                        backoffMs *= 2
                        if backoffMs > conn.maxBackoffMs {
                                backoffMs = conn.maxBackoffMs
                        }
                        continue
                }

                // Reset backoff on successful connection
                backoffMs = conn.baseBackoffMs

                // Start reading messages
                conn.ReadLoop(s.ctx)

                // If we got here, the connection dropped
                select {
                case <-s.ctx.Done():
                        return
                case <-time.After(time.Duration(backoffMs) * time.Millisecond):
                }

                backoffMs *= 2
                if backoffMs > conn.maxBackoffMs {
                        backoffMs = conn.maxBackoffMs
                }
        }
}

// DisconnectExchange disconnects all streams for an exchange.
func (s *ConnectorService) DisconnectExchange(exchange string) {
        s.mu.Lock()
        defer s.mu.Unlock()

        for key, conn := range s.connections {
                if conn.exchange == exchange {
                        conn.Disconnect()
                        delete(s.connections, key)
                }
        }

        s.logger.Info("disconnected exchange", zap.String("exchange", exchange))
}

// GetStatus returns the status of all connections.
func (s *ConnectorService) GetStatus() map[string]*types.ExchangeConnection {
        s.mu.RLock()
        defer s.mu.RUnlock()

        result := make(map[string]*types.ExchangeConnection)
        for key, conn := range s.connections {
                result[key] = conn.GetStatus()
        }
        return result
}

// Shutdown gracefully stops all connections and the Kafka producer.
func (s *ConnectorService) Shutdown() {
        s.logger.Info("shutting down connector service")
        s.cancel()

        s.mu.Lock()
        defer s.mu.Unlock()

        for key, conn := range s.connections {
                conn.Disconnect()
                delete(s.connections, key)
        }

        if s.producer != nil {
                s.producer.Close()
        }

        s.logger.Info("connector service stopped")
}

// SendHeartbeat publishes heartbeat messages periodically.
func (s *ConnectorService) SendHeartbeat() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()

        for {
                select {
                case <-s.ctx.Done():
                        return
                case <-ticker.C:
                        heartbeat := map[string]interface{}{
                                "service":   "connector",
                                "timestamp": time.Now().UnixMilli(),
                                "connections": func() int {
                                        s.mu.RLock()
                                        defer s.mu.RUnlock()
                                        return len(s.connections)
                                }(),
                        }
                        if s.producer != nil {
                                if err := s.producer.Publish(TopicHeartbeat, []byte("connector"), heartbeat); err != nil {
                                        s.logger.Error("failed to publish heartbeat", zap.Error(err))
                                }
                        }
                }
        }
}

func strtolower(s string) string {
        result := make([]byte, len(s))
        for i, c := range s {
                if c >= 'A' && c <= 'Z' {
                        result[i] = byte(c + 32)
                } else {
                        result[i] = byte(c)
                }
        }
        return string(result)
}

func symbolToOKXInstID(symbol string) string {
        // Convert BTCUSDT -> BTC-USDT
        if len(symbol) > 4 && symbol[len(symbol)-4:] == "USDT" {
                return symbol[:len(symbol)-4] + "-USDT"
        }
        return symbol
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
        cfg := config.LoadConnectorConfig()
        if err := cfg.Validate(); err != nil {
                logger.Fatal("invalid configuration", zap.Error(err))
        }

        logger.Info("starting connector service",
                zap.String("port", cfg.Port),
                zap.Strings("exchanges", cfg.Exchanges),
                zap.Strings("symbols", cfg.Symbols),
                zap.String("redpanda_brokers", cfg.RedpandaBrokers),
        )

        // Create connector service
        service, err := NewConnectorService(cfg, logger)
        if err != nil {
                logger.Fatal("failed to create connector service", zap.Error(err))
        }

        // Connect to all exchanges
        service.ConnectAll()

        // Start heartbeat publisher
        go service.SendHeartbeat()

        // Set up HTTP server
        gin.SetMode(gin.ReleaseMode)
        router := gin.New()
        router.Use(gin.Recovery())

        // Health check endpoint
        router.GET("/health", func(c *gin.Context) {
                statuses := service.GetStatus()
                allConnected := true
                for _, s := range statuses {
                        if s.Status != "connected" {
                                allConnected = false
                                break
                        }
                }
                overall := "healthy"
                if !allConnected {
                        overall = "degraded"
                }
                c.JSON(http.StatusOK, gin.H{
                        "status":  overall,
                        "service": "connector",
                        "connections": len(statuses),
                })
        })

        // Connection status endpoint
        router.GET("/connector/status", func(c *gin.Context) {
                c.JSON(http.StatusOK, gin.H{
                        "exchanges": service.GetStatus(),
                })
        })

        // Connect to an exchange
        router.POST("/connector/connect/:exchange", func(c *gin.Context) {
                exchange := c.Param("exchange")
                service.ConnectExchange(exchange)
                c.JSON(http.StatusOK, gin.H{
                        "message": fmt.Sprintf("Connecting to %s", exchange),
                })
        })

        // Disconnect from an exchange
        router.POST("/connector/disconnect/:exchange", func(c *gin.Context) {
                exchange := c.Param("exchange")
                service.DisconnectExchange(exchange)
                c.JSON(http.StatusOK, gin.H{
                        "message": fmt.Sprintf("Disconnected from %s", exchange),
                })
        })

        // Symbols configuration endpoint
        router.GET("/connector/symbols", func(c *gin.Context) {
                c.JSON(http.StatusOK, gin.H{
                        "symbols":   cfg.Symbols,
                        "exchanges": cfg.Exchanges,
                })
        })

        // Kafka producer stats endpoint
        router.GET("/connector/kafka/stats", func(c *gin.Context) {
                if service.producer != nil {
                        msgCount, errCount := service.producer.Stats()
                        c.JSON(http.StatusOK, gin.H{
                                "messages_sent": msgCount,
                                "errors":        errCount,
                        })
                } else {
                        c.JSON(http.StatusOK, gin.H{
                                "status": "not_connected",
                        })
                }
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

        // Wait for shutdown signal
        quit := make(chan os.Signal, 1)
        signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
        sig := <-quit
        logger.Info("received shutdown signal", zap.String("signal", sig.String()))

        // Graceful shutdown
        service.Shutdown()

        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer cancel()

        if err := srv.Shutdown(ctx); err != nil {
                logger.Error("HTTP server forced shutdown", zap.Error(err))
        }

        logger.Info("connector service stopped")
}

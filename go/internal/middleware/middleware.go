// Package middleware provides shared HTTP middleware for ACMS Go services.
package middleware

import (
        "bytes"
        "context"
        "crypto/rand"
        "crypto/subtle"
        "encoding/json"
        "fmt"
        "io"
        "net/http"
        "strings"
        "sync"
        "time"

        "github.com/gin-gonic/gin"
        "github.com/golang-jwt/jwt/v5"
        "github.com/redis/go-redis/v9"
        "go.uber.org/zap"
)

// JWTAuthMiddleware validates JWT tokens from the Authorization header.
// It parses and verifies the token using the provided secret, then sets
// user claims in the gin context.
func JWTAuthMiddleware(secret string, logger *zap.Logger) gin.HandlerFunc {
        return func(c *gin.Context) {
                authHeader := c.GetHeader("Authorization")
                if authHeader == "" {
                        c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                                "error": "missing authorization header",
                        })
                        return
                }

                tokenStr := strings.TrimPrefix(authHeader, "Bearer ")
                if tokenStr == authHeader {
                        c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                                "error": "invalid token format, expected Bearer <token>",
                        })
                        return
                }

                token, err := jwt.Parse(tokenStr, func(token *jwt.Token) (interface{}, error) {
                        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
                                return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
                        }
                        return []byte(secret), nil
                })

                if err != nil {
                        logger.Debug("JWT validation failed", zap.Error(err))
                        c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                                "error": "invalid or expired token",
                        })
                        return
                }

                if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
                        c.Set("user_id", claims["sub"])
                        c.Set("user_role", claims["role"])
                        c.Set("token_expiry", claims["exp"])
                        c.Next()
                } else {
                        c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                                "error": "invalid token claims",
                        })
                        return
                }
        }
}

// OptionalJWTAuth validates JWT if present but does not require it.
func OptionalJWTAuth(secret string, logger *zap.Logger) gin.HandlerFunc {
        return func(c *gin.Context) {
                authHeader := c.GetHeader("Authorization")
                if authHeader == "" {
                        c.Set("authenticated", false)
                        c.Next()
                        return
                }

                tokenStr := strings.TrimPrefix(authHeader, "Bearer ")
                if tokenStr == authHeader {
                        c.Set("authenticated", false)
                        c.Next()
                        return
                }

                token, err := jwt.Parse(tokenStr, func(token *jwt.Token) (interface{}, error) {
                        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
                                return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
                        }
                        return []byte(secret), nil
                })

                if err == nil {
                        if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
                                c.Set("user_id", claims["sub"])
                                c.Set("user_role", claims["role"])
                                c.Set("authenticated", true)
                        }
                } else {
                        c.Set("authenticated", false)
                }
                c.Next()
        }
}

// RedisRateLimiter implements a Redis-backed token bucket rate limiter.
// Rate limiting state persists across restarts since it is stored in Redis.
type RedisRateLimiter struct {
        client   *redis.Client
        limit    int
        window   time.Duration
        prefix   string
        logger   *zap.Logger
        localMu  sync.Mutex
        local    map[string]*localBucket
}

type localBucket struct {
        tokens    int
        lastCheck time.Time
}

// NewRedisRateLimiter creates a new Redis-backed rate limiter.
func NewRedisRateLimiter(client *redis.Client, limit int, window time.Duration, logger *zap.Logger) *RedisRateLimiter {
        return &RedisRateLimiter{
                client: client,
                limit:  limit,
                window: window,
                prefix: "acms:ratelimit:",
                logger: logger,
                local:  make(map[string]*localBucket),
        }
}

// Allow checks if a request from the given key is allowed.
// Falls back to in-memory rate limiting if Redis is unavailable.
func (rl *RedisRateLimiter) Allow(key string) bool {
        ctx := context.Background()
        redisKey := rl.prefix + key

        // Try Redis first
        if rl.client != nil {
                count, err := rl.client.Incr(ctx, redisKey).Result()
                if err != nil {
                        rl.logger.Warn("Redis rate limit check failed, falling back to local", zap.Error(err))
                        return rl.allowLocal(key)
                }
                if count == 1 {
                        rl.client.Expire(ctx, redisKey, rl.window)
                }
                return count <= int64(rl.limit)
        }

        return rl.allowLocal(key)
}

func (rl *RedisRateLimiter) allowLocal(key string) bool {
        rl.localMu.Lock()
        defer rl.localMu.Unlock()

        now := time.Now()
        bucket, exists := rl.local[key]

        if !exists || now.Sub(bucket.lastCheck) > rl.window {
                rl.local[key] = &localBucket{tokens: rl.limit - 1, lastCheck: now}
                return true
        }

        if bucket.tokens <= 0 {
                return false
        }

        bucket.tokens--
        bucket.lastCheck = now
        return true
}

// RateLimitMiddleware creates a rate limiting middleware using Redis.
func RateLimitMiddleware(limiter *RedisRateLimiter) gin.HandlerFunc {
        return func(c *gin.Context) {
                ip := c.ClientIP()
                if !limiter.Allow(ip) {
                        c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
                                "error": "rate limit exceeded",
                        })
                        return
                }
                c.Next()
        }
}

// RequestLoggerMiddleware logs all requests using zap structured logging.
func RequestLoggerMiddleware(logger *zap.Logger) gin.HandlerFunc {
        return func(c *gin.Context) {
                start := time.Now()
                path := c.Request.URL.Path
                query := c.Request.URL.RawQuery

                // Capture response body size
                blw := &bodyLogWriter{body: bytes.NewBufferString(""), ResponseWriter: c.Writer}
                c.Writer = blw

                c.Next()

                latency := time.Since(start)
                statusCode := c.Writer.Status()

                fields := []zap.Field{
                        zap.String("method", c.Request.Method),
                        zap.String("path", path),
                        zap.String("query", query),
                        zap.Int("status", statusCode),
                        zap.Duration("latency", latency),
                        zap.String("client_ip", c.ClientIP()),
                        zap.String("user_agent", c.Request.UserAgent()),
                }

                if userID, exists := c.Get("user_id"); exists {
                        fields = append(fields, zap.Any("user_id", userID))
                }

                if statusCode >= 500 {
                        logger.Error("request completed", fields...)
                } else if statusCode >= 400 {
                        logger.Warn("request completed", fields...)
                } else {
                        logger.Info("request completed", fields...)
                }
        }
}

type bodyLogWriter struct {
        gin.ResponseWriter
        body *bytes.Buffer
}

func (w bodyLogWriter) Write(b []byte) (int, error) {
        w.body.Write(b)
        return w.ResponseWriter.Write(b)
}

// CORSMiddleware handles Cross-Origin Resource Sharing.
func CORSMiddleware(allowedHosts []string) gin.HandlerFunc {
        return func(c *gin.Context) {
                origin := c.Request.Header.Get("Origin")
                for _, allowed := range allowedHosts {
                        if allowed == "*" || allowed == origin {
                                if origin != "" {
                                        c.Header("Access-Control-Allow-Origin", origin)
                                } else if allowed == "*" {
                                        c.Header("Access-Control-Allow-Origin", "*")
                                }
                                break
                        }
                }

                c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
                c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Authorization, X-Request-ID, X-API-Version")
                c.Header("Access-Control-Expose-Headers", "Content-Length, X-Request-ID")
                c.Header("Access-Control-Max-Age", "86400")
                c.Header("Vary", "Origin")

                if c.Request.Method == "OPTIONS" {
                        c.AbortWithStatus(http.StatusNoContent)
                        return
                }

                c.Next()
        }
}

// IPFilterMiddleware blocks or allows IPs based on whitelist/blacklist.
func IPFilterMiddleware(blacklist, whitelist []string, logger *zap.Logger) gin.HandlerFunc {
        whitelistSet := make(map[string]bool, len(whitelist))
        for _, ip := range whitelist {
                whitelistSet[ip] = true
        }
        blacklistSet := make(map[string]bool, len(blacklist))
        for _, ip := range blacklist {
                blacklistSet[ip] = true
        }

        hasWhitelist := len(whitelist) > 0

        return func(c *gin.Context) {
                ip := c.ClientIP()

                if blacklistSet[ip] {
                        logger.Warn("blocked IP from blacklist", zap.String("ip", ip))
                        c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "access denied"})
                        return
                }

                if hasWhitelist && !whitelistSet[ip] {
                        logger.Warn("blocked IP not in whitelist", zap.String("ip", ip))
                        c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "access denied"})
                        return
                }

                c.Next()
        }
}

// RequestSizeLimitMiddleware rejects requests exceeding the size limit.
func RequestSizeLimitMiddleware(maxBytes int64) gin.HandlerFunc {
        return func(c *gin.Context) {
                if c.Request.ContentLength > maxBytes {
                        c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{
                                "error": fmt.Sprintf("request body too large, max %d bytes", maxBytes),
                        })
                        return
                }
                c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBytes)
                c.Next()
        }
}

// APIKeyAuthMiddleware validates API key in the X-API-Key header.
func APIKeyAuthMiddleware(validKeys []string, logger *zap.Logger) gin.HandlerFunc {
        keySet := make(map[string]bool, len(validKeys))
        for _, k := range validKeys {
                keySet[k] = true
        }

        return func(c *gin.Context) {
                apiKey := c.GetHeader("X-API-Key")
                if apiKey == "" {
                        c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "missing API key"})
                        return
                }

                // Constant-time comparison to prevent timing attacks
                found := false
                for _, validKey := range validKeys {
                        if subtle.ConstantTimeCompare([]byte(apiKey), []byte(validKey)) == 1 {
                                found = true
                                break
                        }
                }

                if !found {
                        logger.Warn("invalid API key attempt", zap.String("client_ip", c.ClientIP()))
                        c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "invalid API key"})
                        return
                }

                c.Set("api_key", apiKey)
                c.Next()
        }
}

// RecoveryMiddleware recovers from panics and returns a 500 error.
func RecoveryMiddleware(logger *zap.Logger) gin.HandlerFunc {
        return func(c *gin.Context) {
                defer func() {
                        if err := recover(); err != nil {
                                logger.Error("panic recovered",
                                        zap.Any("error", err),
                                        zap.String("path", c.Request.URL.Path),
                                        zap.String("method", c.Request.Method),
                                )
                                c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
                                        "error": "internal server error",
                                })
                        }
                }()
                c.Next()
        }
}

// VersionMiddleware adds API version to response headers.
func VersionMiddleware(version string) gin.HandlerFunc {
        return func(c *gin.Context) {
                c.Header("X-API-Version", version)
                c.Next()
        }
}

// RequestIDMiddleware adds a unique request ID to each request.
func RequestIDMiddleware() gin.HandlerFunc {
        return func(c *gin.Context) {
                requestID := c.GetHeader("X-Request-ID")
                if requestID == "" {
                        requestID = fmt.Sprintf("%d-%s", time.Now().UnixNano(), randomString(8))
                }
                c.Set("request_id", requestID)
                c.Header("X-Request-ID", requestID)
                c.Next()
        }
}

func randomString(length int) string {
        const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        b := make([]byte, length)
        _, err := rand.Read(b)
        if err != nil {
                // Fallback to timestamp-based
                for i := range b {
                        b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
                }
                return string(b)
        }
        for i := range b {
                b[i] = charset[int(b[i])%len(charset)]
        }
        return string(b)
}

// CircuitBreaker implements the circuit breaker pattern for backend calls.
type CircuitBreaker struct {
        mu              sync.Mutex
        state           string // "closed", "open", "half-open"
        failureCount    int
        failureThreshold int
        resetTimeout    time.Duration
        lastFailure     time.Time
        halfOpenSuccess int
        halfOpenMax     int
        logger          *zap.Logger
}

// NewCircuitBreaker creates a new circuit breaker.
func NewCircuitBreaker(failureThreshold int, resetTimeout time.Duration, logger *zap.Logger) *CircuitBreaker {
        return &CircuitBreaker{
                state:           "closed",
                failureThreshold: failureThreshold,
                resetTimeout:    resetTimeout,
                halfOpenMax:     3,
                logger:          logger,
        }
}

// Allow checks if the circuit breaker allows a request through.
func (cb *CircuitBreaker) Allow() bool {
        cb.mu.Lock()
        defer cb.mu.Unlock()

        switch cb.state {
        case "closed":
                return true
        case "open":
                if time.Since(cb.lastFailure) > cb.resetTimeout {
                        cb.state = "half-open"
                        cb.halfOpenSuccess = 0
                        cb.logger.Info("circuit breaker transitioning to half-open")
                        return true
                }
                return false
        case "half-open":
                return cb.halfOpenSuccess < cb.halfOpenMax
        default:
                return false
        }
}

// RecordSuccess records a successful call.
func (cb *CircuitBreaker) RecordSuccess() {
        cb.mu.Lock()
        defer cb.mu.Unlock()

        if cb.state == "half-open" {
                cb.halfOpenSuccess++
                if cb.halfOpenSuccess >= cb.halfOpenMax {
                        cb.state = "closed"
                        cb.failureCount = 0
                        cb.logger.Info("circuit breaker closed after successful recovery")
                }
        } else if cb.state == "closed" {
                cb.failureCount = 0
        }
}

// RecordFailure records a failed call.
func (cb *CircuitBreaker) RecordFailure() {
        cb.mu.Lock()
        defer cb.mu.Unlock()

        cb.failureCount++
        cb.lastFailure = time.Now()

        if cb.state == "half-open" {
                cb.state = "open"
                cb.logger.Warn("circuit breaker opened after failure in half-open state")
                return
        }

        if cb.failureCount >= cb.failureThreshold {
                cb.state = "open"
                cb.logger.Warn("circuit breaker opened due to failure threshold",
                        zap.Int("failure_count", cb.failureCount),
                        zap.Int("threshold", cb.failureThreshold),
                )
        }
}

// State returns the current state of the circuit breaker.
func (cb *CircuitBreaker) State() string {
        cb.mu.Lock()
        defer cb.mu.Unlock()
        return cb.state
}

// ProxyWithCircuitBreaker wraps a reverse proxy handler with circuit breaker logic.
func ProxyWithCircuitBreaker(target string, cb *CircuitBreaker, logger *zap.Logger) gin.HandlerFunc {
        return func(c *gin.Context) {
                if !cb.Allow() {
                        c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
                                "error":  "service temporarily unavailable",
                                "state":  cb.State(),
                        })
                        return
                }

                // Read request body for potential retry
                var bodyBytes []byte
                if c.Request.Body != nil {
                        bodyBytes, _ = io.ReadAll(c.Request.Body)
                        c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
                }

                // Make the proxied request
                urlStr := target + c.Request.URL.Path
                if c.Request.URL.RawQuery != "" {
                        urlStr += "?" + c.Request.URL.RawQuery
                }

                proxyReq, err := http.NewRequest(c.Request.Method, urlStr, bytes.NewBuffer(bodyBytes))
                if err != nil {
                        cb.RecordFailure()
                        c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "proxy request failed"})
                        return
                }

                // Copy headers
                for key, values := range c.Request.Header {
                        for _, value := range values {
                                proxyReq.Header.Add(key, value)
                        }
                }

                client := &http.Client{Timeout: 30 * time.Second}
                resp, err := client.Do(proxyReq)
                if err != nil {
                        cb.RecordFailure()
                        logger.Error("proxy request failed", zap.Error(err), zap.String("target", target))
                        c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "upstream service unavailable"})
                        return
                }
                defer resp.Body.Close()

                if resp.StatusCode >= 500 {
                        cb.RecordFailure()
                } else {
                        cb.RecordSuccess()
                }

                // Copy response headers
                for key, values := range resp.Header {
                        for _, value := range values {
                                c.Writer.Header().Add(key, value)
                        }
                }

                c.Writer.WriteHeader(resp.StatusCode)
                respBody, _ := io.ReadAll(resp.Body)
                var jsonBody map[string]interface{}
                if json.Unmarshal(respBody, &jsonBody) == nil {
                        c.JSON(resp.StatusCode, jsonBody)
                } else {
                        c.Writer.Write(respBody)
                }
        }
}

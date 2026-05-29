# Go Services Multi-Stage Dockerfile
# Supports building all 3 services: gateway, monitor, connector
# Usage: docker build --build-arg SERVICE=gateway -t acms-gateway .

FROM golang:1.22-alpine AS builder

ARG SERVICE

WORKDIR /build

# Install build dependencies for CGO (needed for confluent-kafka-go)
RUN apk add --no-cache gcc musl-dev pkgconf

# Copy module files first for layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY cmd/ ./cmd/
COPY internal/ ./internal/

# Build the specified service with CGO enabled for kafka
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-s -w" -o /service ./cmd/${SERVICE}

# Runtime stage - minimal image
FROM alpine:3.19

RUN apk --no-cache add ca-certificates tzdata

# Create non-root user
RUN addgroup -S acms && adduser -S -G acms acms

COPY --from=builder /service /service

# Create data directories
RUN mkdir -p /data/acms/logs && chown -R acms:acms /data

USER acms

# Port is configured via environment variable, expose common ports
EXPOSE 8080 8081 8082

ENTRYPOINT ["/service"]

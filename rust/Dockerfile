# Rust Core Dockerfile
FROM rust:1.77-slim AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

RUN cargo build --release

# Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/acms_core /usr/local/bin/

CMD ["acms_core"]

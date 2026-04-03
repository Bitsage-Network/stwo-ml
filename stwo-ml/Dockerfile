# ObelyZK Prove Server — Production Docker Image
#
# Build:
#   docker build -t obelyzk/prove-server .
#
# Run (CPU only):
#   docker run -p 8080:8080 -v /models:/models obelyzk/prove-server
#
# Run (with NVIDIA GPU):
#   docker run --gpus all -p 8080:8080 -v /models:/models obelyzk/prove-server
#
# API:
#   POST /api/v1/models      — Load a model
#   POST /api/v1/infer        — Provable inference (input → output + proof)
#   GET  /api/v1/verify/:hash — Verify a proof
#   GET  /api/v1/proofs       — List proven inferences
#   GET  /health              — Health check

# Stage 1: Build
FROM rust:1.87-bookworm AS builder

# Install nightly toolchain matching STWO
RUN rustup toolchain install nightly-2025-07-14 && \
    rustup default nightly-2025-07-14

WORKDIR /build

# Copy workspace
COPY . .

# Build release binary with server features
# CPU-only by default; for GPU, add --features cuda-runtime
RUN cargo build --release --bin prove-server \
    --features "server,model-loading,safetensors,binary-proof" \
    2>&1 | tail -5

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/prove-server /usr/local/bin/prove-server

# Default model directory
RUN mkdir -p /models /data/proofs

ENV BIND_ADDR=0.0.0.0:8080
ENV RUST_LOG=info

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["prove-server"]

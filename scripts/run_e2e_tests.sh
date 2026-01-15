#!/bin/bash
# =============================================================================
# Umi End-to-End Test Runner
# =============================================================================
#
# This script runs comprehensive end-to-end tests for Umi with real providers.
#
# Prerequisites:
#   - Rust toolchain installed
#   - ANTHROPIC_API_KEY environment variable (optional, for Anthropic tests)
#   - OPENAI_API_KEY environment variable (optional, for OpenAI tests)
#
# Usage:
#   ./scripts/run_e2e_tests.sh           # Run all available tests
#   ./scripts/run_e2e_tests.sh --sim     # Run simulation tests only (no API keys needed)
#   ./scripts/run_e2e_tests.sh --full    # Run full tests (requires API keys)
#
# Environment Variables:
#   SKIP_ANTHROPIC=1    Skip Anthropic tests
#   SKIP_OPENAI=1       Skip OpenAI tests
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Umi Memory - E2E Test Runner                     ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
SIM_ONLY=false
FULL_TESTS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --sim)
            SIM_ONLY=true
            shift
            ;;
        --full)
            FULL_TESTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--sim|--full]"
            exit 1
            ;;
    esac
done

# Check for API keys
echo -e "${YELLOW}Checking environment...${NC}"
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "  ANTHROPIC_API_KEY: ${GREEN}Set${NC}"
else
    echo -e "  ANTHROPIC_API_KEY: ${RED}Not set${NC}"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "  OPENAI_API_KEY: ${GREEN}Set${NC}"
else
    echo -e "  OPENAI_API_KEY: ${RED}Not set${NC}"
fi
echo ""

# Navigate to workspace root
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo ""

# =============================================================================
# Step 1: Run Unit Tests
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: Running Unit Tests${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Running cargo test with all features..."
cargo test --all-features -p umi-memory --lib -- --test-threads=4

# =============================================================================
# Step 2: Run Integration Tests (Simulation)
# =============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Running Integration Tests (Simulation)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Running integration tests..."
cargo test --all-features -p umi-memory --test '*' -- --test-threads=4

if [ "$SIM_ONLY" = true ]; then
    echo ""
    echo -e "${GREEN}Simulation tests completed successfully!${NC}"
    echo "Run with --full flag to also run real provider tests (requires API keys)"
    exit 0
fi

# =============================================================================
# Step 3: Run Full E2E Test (Real Providers)
# =============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Running Full E2E Test (Real Providers)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: No API keys set. Running with simulation providers only.${NC}"
    echo "Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY to run real provider tests."
    echo ""
fi

echo "Running full E2E test..."
cargo run --example full_e2e_test --all-features

# =============================================================================
# Step 4: Run Individual Provider Tests (if API keys available)
# =============================================================================
if [ -n "$ANTHROPIC_API_KEY" ] && [ "$SKIP_ANTHROPIC" != "1" ]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 4a: Running Anthropic Provider Test${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    cargo run --example test_anthropic --features anthropic
fi

if [ -n "$OPENAI_API_KEY" ] && [ "$SKIP_OPENAI" != "1" ]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 4b: Running OpenAI Provider Test${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    cargo run --example test_openai --features openai,embedding-openai
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    All Tests Completed!                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ -n "$ANTHROPIC_API_KEY" ] && [ -n "$OPENAI_API_KEY" ]; then
    echo -e "${GREEN}Full test suite executed with real LLM providers!${NC}"
else
    echo -e "${YELLOW}Partial test suite executed. For full testing:${NC}"
    [ -z "$ANTHROPIC_API_KEY" ] && echo "  - Set ANTHROPIC_API_KEY for Anthropic tests"
    [ -z "$OPENAI_API_KEY" ] && echo "  - Set OPENAI_API_KEY for OpenAI tests"
fi

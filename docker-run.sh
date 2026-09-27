#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "   Obscuro Erebus 32B Foundation Model - Docker Launcher (Linux/Mac)"
echo "=========================================================================="
echo ""
echo "  [1] Full 32B Pipeline Run (install datasets + GPU/CPU train + save)"
echo "  [2] Start Autonomous Worker Daemon (continuous: learn + train + repeat)"
echo "  [3] Run Chromium Self-Browser Learning Session Only"
echo "  [4] GPU + Hardware Diagnostics Check (syscheck)"
echo "  [5] Stop All Running Containers"
echo "  [6] View Live Worker Logs"
echo "  [7] Rebuild Docker Image (after code changes)"
echo ""
echo "=========================================================================="
read -p "Select option (1-7): " choice

case "$choice" in
  1)
    echo ""
    echo "[START] Building and running full 32B foundation model pipeline..."
    docker compose run --rm app fullrun
    ;;
  2)
    echo ""
    echo "[START] Launching autonomous self-training worker daemon..."
    docker compose up -d worker
    echo ""
    echo "Worker is running in background!"
    echo "Monitor live training logs with: docker compose logs -f worker"
    ;;
  3)
    echo ""
    echo "[START] Running autonomous Chromium browser learning session..."
    docker compose run --rm learner
    ;;
  4)
    echo ""
    echo "[START] Running system + GPU/CPU diagnostics..."
    docker compose run --rm app syscheck
    ;;
  5)
    echo ""
    echo "[STOP] Stopping all containers..."
    docker compose down
    ;;
  6)
    echo ""
    echo "[LOGS] Streaming live worker logs (Ctrl+C to stop)..."
    docker compose logs -f worker
    ;;
  7)
    echo ""
    echo "[BUILD] Rebuilding Docker image with latest code changes..."
    docker compose build --no-cache
    echo "[OK] Rebuild complete."
    ;;
  *)
    echo "Invalid choice. Please run the script again and select 1-7."
    ;;
esac

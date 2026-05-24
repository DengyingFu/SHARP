#!/bin/bash

# Start dino_service in the background
echo "Starting dino_service..."
# Enter ambiguity directory to run the service as requested
(cd ambiguity && python utils/dino_service.py) &
DINO_PID=$!

# Start dual_model_services in the background
echo "Starting dual_model_services..."
python start_dual_model_services.py &
DUAL_PID=$!

# Function to handle script kill (Ctrl+C)
cleanup() {
    echo "Stopping services..."
    kill $DINO_PID
    kill $DUAL_PID
    exit
}

# Trap SIGINT (Ctrl+C) to run cleanup
trap cleanup SIGINT

echo "Services started with PIDs: dino_service=$DINO_PID, dual_model=$DUAL_PID"
echo "Press Ctrl+C to stop all services."

# Wait for processes to finish
wait

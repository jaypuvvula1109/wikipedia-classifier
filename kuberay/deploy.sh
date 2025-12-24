#!/bin/bash
# =============================================================================
# Step 6.3: Build and Deploy Application
# =============================================================================

set -e

CLUSTER_NAME="wikipedia-classifier"
IMAGE_NAME="wikipedia-classifier:latest"

echo "============================================================"
echo "STEP 6.3: Build and Deploy Application"
echo "============================================================"

# Check Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found. Run from project root."
    exit 1
fi

# Check model exists
if [ ! -f "../models/text_classifier.pkl" ]; then
    echo "ERROR: Model not found. Run training first."
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Load into Kind
echo ""
echo "Loading image into Kind cluster..."
kind load docker-image $IMAGE_NAME --name $CLUSTER_NAME

# Deploy RayService
echo ""
echo "Deploying RayService..."
kubectl apply -f kuberay/ray-service.yaml
#!/bin/bash
# =============================================================================
# Step 6.1: Create Local Kubernetes Cluster with Kind
# =============================================================================

set -e

echo "============================================================"
echo "STEP 6.1: Create Local Kubernetes Cluster"
echo "============================================================"

CLUSTER_NAME="wikipedia-classifier"

# Check if Kind is installed
if ! command -v kind &> /dev/null; then
    echo "ERROR: Kind is not installed."
    echo "Install with: brew install kind"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running. Please start Docker."
    exit 1
fi

# Delete existing cluster if it exists
if kind get clusters | grep -q "$CLUSTER_NAME"; then
    echo "Deleting existing cluster '$CLUSTER_NAME'..."
    kind delete cluster --name "$CLUSTER_NAME"
fi

# Create Kind cluster
echo ""
echo "Creating Kind cluster '$CLUSTER_NAME'..."
kind create cluster --name "$CLUSTER_NAME" --config - <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
EOF

# Verify
echo ""
kubectl cluster-info --context kind-$CLUSTER_NAME
kubectl get nodes

echo ""
echo "✓ Kind cluster '$CLUSTER_NAME' created!"
echo "Next: ./04_kuberay/install_kuberay.sh"
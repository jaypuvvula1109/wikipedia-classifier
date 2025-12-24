
#!/bin/bash
# =============================================================================
# Step 6.2: Install KubeRay Operator
# =============================================================================

set -e

echo "============================================================"
echo "STEP 6.2: Install KubeRay Operator"
echo "============================================================"

# Check Helm
if ! command -v helm &> /dev/null; then
    echo "ERROR: Helm is not installed."
    echo "Install with: brew install helm"
    exit 1
fi

# Add KubeRay Helm repo
echo "Adding KubeRay Helm repository..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install KubeRay operator
echo ""
echo "Installing KubeRay operator..."
helm install kuberay-operator kuberay/kuberay-operator \
    --version 1.2.2 \
    --create-namespace \
    --namespace kuberay-system

# Wait for operator
echo ""
echo "Waiting for KubeRay operator to be ready..."
kubectl wait --for=condition=available --timeout=120s \
    deployment/kuberay-operator -n kuberay-system

kubectl get pods -n kuberay-system
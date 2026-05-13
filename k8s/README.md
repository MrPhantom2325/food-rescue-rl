# Kubernetes manifests

These manifests describe how the Food Rescue RL system would run on a real Kubernetes cluster. They are written but **not deployed** as part of Phase 1 — the local-dev story is `docker compose` (see `../docker-compose.yml`).

The intent here is to demonstrate the GitOps principle of "infrastructure as code, version-controlled, declarative": every piece of cluster state lives in this folder, and a GitOps controller like ArgoCD or Flux can sync `k8s/` to a cluster automatically.

## What's here

| File                | Purpose                                                 |
| ------------------- | ------------------------------------------------------- |
| `00-namespace.yaml` | Dedicated `food-rescue` namespace                       |
| `10-mlflow.yaml`    | MLflow tracking server (Deployment + Service + PVC)     |
| `20-api.yaml`       | FastAPI prediction service (Deployment + Service + PVC) |
| `30-train-job.yaml` | One-shot training Job (writes policy to shared PVC)     |

## How to apply (if you do have a cluster)

```bash
# Build and load images into the cluster (kind/minikube)
docker build -t food-rescue-train:latest -f Dockerfile.train .
docker build -t food-rescue-serve:latest -f Dockerfile.serve .

# For kind:
kind load docker-image food-rescue-train:latest
kind load docker-image food-rescue-serve:latest

# Apply in order
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/10-mlflow.yaml
kubectl apply -f k8s/20-api.yaml

# Wait for API to come up
kubectl -n food-rescue wait --for=condition=available --timeout=120s deployment/api

# Run training as a Job (writes policy into the shared PVC)
kubectl apply -f k8s/30-train-job.yaml
kubectl -n food-rescue logs -l app=train -f

# Port-forward the API
kubectl -n food-rescue port-forward svc/api 8000:80
# → http://localhost:8000/health
```

## GitOps story

In a production setup you'd add an ArgoCD `Application` pointing at this folder:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: food-rescue
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/<you>/food-rescue-rl.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: food-rescue
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

Then every push to `main` that changes anything under `k8s/` is automatically reconciled into the cluster. That's GitOps.

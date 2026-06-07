# Smoke-Test

Checkt End-to-End: API-Server, Storage, Pod-Scheduling, Service-Networking, Volume-Mount.

```bash
kubectl apply -f infrastructure/k8s/smoke-test/
kubectl rollout status deploy/smoke-nginx -n smoke --timeout=60s
kubectl port-forward -n smoke svc/smoke-nginx 18080:80 &
sleep 2 && curl -s http://localhost:18080/ && kill %1
```

Erwartete Ausgabe: `k3d smoke ok @ <timestamp>`.

Wieder weg: `kubectl delete ns smoke`.

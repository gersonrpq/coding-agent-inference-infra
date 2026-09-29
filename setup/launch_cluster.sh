
source .env 



echo "=================== Deploying Namespaces and Ksustomization... ==================="
kubectl apply -f cluster/namespace.yaml
#kubectl apply -f cluster/kustomization.yaml

echo "=================== Deploying LiteLLM... ==================="
envsubst < cluster/litellm/secret.yaml | kubectl apply -f -

kubectl create configmap litellm-security \
  -n gpu-serving \
  --from-file=security_inspect.py=control/inspect.py \
  --dry-run=client \
  -o yaml | kubectl apply -f -

kubectl apply -f cluster/litellm/config.yaml
kubectl apply -f cluster/litellm/deployment.yaml
kubectl apply -f cluster/litellm/service.yaml

echo "=================== Deploying SGLang... ==================="
kubectl apply -f cluster/sglang/config.yaml
kubectl apply -f cluster/sglang/services.yaml
kubectl apply -f cluster/sglang/worker-0.yaml
kubectl apply -f cluster/sglang/worker-1.yaml

echo "=================== Deploying Monitoring Prometheus... ==================="
kubectl apply -f cluster/monitoring/prometheus-config.yaml
kubectl apply -f cluster/monitoring/prometheus.yaml

echo "=================== Deploying Monitoring Grafana... ==================="
kubectl apply -f cluster/monitoring/grafana.yaml
kubectl apply -f cluster/monitoring/grafana-datasource.yaml
kubectl apply -f cluster/monitoring/services.yaml

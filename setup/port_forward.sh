#!/bin/bash

kubectl port-forward -n gpu-serving service/prometheus 9090:9090 &
kubectl port-forward -n gpu-serving service/grafana 3000:3000 &
kubectl port-forward -n gpu-serving service/litellm 4000:4000 &
kubectl port-forward -n gpu-serving service/sglang-worker-0 30000:30000 &
kubectl port-forward -n gpu-serving service/sglang-worker-1 30001:30001 &

wait

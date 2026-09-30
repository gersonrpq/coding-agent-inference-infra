echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -x /usr/local/bin/k3s && ! -x /usr/bin/k3s ]]; then
  echo "== k3s (nvidia default runtime) =="
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--write-kubeconfig-mode 644 --default-runtime nvidia" sh -
else
  echo "== k3s already installed =="
fi

export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
if [[ ! -r "$KUBECONFIG" ]]; then
  echo "Cannot read $KUBECONFIG — re-run as a user who can, or sudo chmod 644."
  exit 1
fi

echo "== NVIDIA DEAMONSET =="
kubectl apply -f cluster/nvidia/config.yaml
kubectl apply -f cluster/nvidia/device-plugin.yaml

# PyTorch Conference 2025 - vLLM ROCm Benchmarking

This repository contains tools and configurations for benchmarking vLLM
performance on AMD GPUs using Kubernetes with Model Kernel caching
provided by GKM capabilities.

## Overview

The project demonstrates performance comparisons between different vLLM
versions running on AMD ROCm, featuring:

- Containerized vLLM deployments with ROCm support
- Kubernetes-based orchestration with AMD GPU allocation
- Model caching using GKM (GPU Kernel Manager) for faster startup times
- Automated benchmarking with throughput and latency measurements

## Repository Structure

```
├── images/                   # dir of Containerfiles
│   ├── Makefile              # Container image build automation
│   ├── vllm-latest/          # Latest vLLM container
│   │   ├── Containerfile.vllm-rocm
│   │   └── entrypoint-vllm.sh
│   └── vllm-old/             # Legacy vLLM container for comparison
│       ├── Containerfile.vllm-rocm-old
│       └── entrypoint-vllm.sh
├── k8s/                      # Kubernetes manifests
│   ├── 00-namespace.yaml     # Namespace setup
│   ├── 01-hf-token.yaml      # HuggingFace token secret
│   ├── 20-llama-cache.yaml   # Model cache CRD
│   ├── 30-llama-rocm-cached-pod.yaml  # Latest vLLM with caching
│   └── 31-llama-rocm-pod.yaml         # Legacy vLLM without caching
└── scripts/
    └── kubeadm-amd.sh        # Kubernetes cluster setup for AMD GPUs
```

## Prerequisites

- Linux system with AMD GPUs and ROCm drivers installed
- Container runtime (Podman or Docker)
- Kubernetes cluster or kubeadm for single-node setup
- GKM deployed in the Kubernetes cluster
- HuggingFace account with API token for model access

## Quick Start

### 1. Set up Kubernetes Cluster with AMD GPU Support

Use the provided script to set up a single-node cluster with AMD GPU support:

```bash
./scripts/kubeadm-amd.sh create
```

#### Clean up the cluster (if needed)

```bash
./scripts/kubeadm-amd.sh cleanup
```

### 2. Build Container Images

Build both vLLM container variants:

```bash
cd images/
make all
```

Or build individually:
```bash
make build-latest  # Latest vLLM version
make build-old     # Legacy vLLM version for comparison
```

> Note: you need to push these image to an image registry so that
they can be used later.

### 3. Deploy to Kubernetes

0. **Deploy GKM**

1. **Create namespace:**

   ```bash
   kubectl apply -f k8s/00-namespace.yaml
   ```

2. **Configure HuggingFace token:**

   ```bash
   # Encode your HF token
   echo -n "your_hf_token_here" | base64

   # Update k8s/01-hf-token.yaml with the encoded token
   kubectl apply -f k8s/01-hf-token.yaml
   ```

3. **Set up model caching (optional):**

   ```bash
   kubectl apply -f k8s/20-llama-cache.yaml
   ```

4. **Deploy vLLM pods:**

   ```bash
   # Deploy latest vLLM with caching
   kubectl apply -f k8s/30-llama-rocm-cached-pod.yaml

   # Deploy legacy vLLM for comparison
   kubectl apply -f k8s/31-llama-rocm-pod.yaml
   ```

## Usage Examples

### Running Benchmarks

The [entrypoint script](images/vllm-latest/entrypoint-vllm.sh) supports multiple modes:

#### 1. Throughput Benchmarking

Set `MODE=benchmark-throughput` in the pod environment to run throughput tests:

```yaml
env:
  - name: MODE
    value: benchmark-throughput
  - name: NUM_PROMPTS
    value: "1000"
  - name: INPUT_LEN
    value: "512"
  - name: OUTPUT_LEN
    value: "256"
```


#### 2. Latency Benchmarking
Set `MODE=benchmark-latency` for latency measurements:

```yaml
env:
  - name: MODE
    value: benchmark-latency
```

### Configuration Options

The [entrypoint scripts](images/vllm-latest/entrypoint-vllm.sh) support extensive configuration via environment variables:

#### Basic Configuration

```yaml
env:
  - name: MODEL
    value: "RedHatAI/Llama-3.1-8B-Instruct"
  - name: PORT
    value: "8000"
  - name: MODE
    value: "serve"  # Options: serve, benchmark-throughput, benchmark-latency
```

#### Benchmark Configuration

```yaml
env:
  - name: INPUT_LEN
    value: "512"
  - name: OUTPUT_LEN
    value: "256"
  - name: NUM_PROMPTS
    value: "1000"
  - name: MAX_BATCH_TOKENS
    value: "8192"
  - name: BENCHMARK_SUMMARY_MODE
    value: "table"  # Options: table, graph, none
```

#### vLLM Options

```yaml
env:
  - name: VLLM_USE_COMPILED_ATTENTION
    value: "1"
  - name: VLLM_COMPILED_ATTENTION_BACKEND
    value: "1"
  - name: VLLM_USE_V1
    value: "1"
  - name: EXTRA_ARGS
    value: >-
      --max-model-len 8192
      --max-num-batched-tokens 8192
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.95
```

## Model Caching with GKM

The [cached pod configuration](k8s/30-llama-rocm-cached-pod.yaml) demonstrates model caching for faster startup times:

```yaml
volumeMounts:
  - name: model-cache-volume
    mountPath: /home/vllm/.cache/vllm/
  - name: model-hf-volume
    mountPath: /home/vllm/.cache/huggingface/

volumes:
  - name: model-hf-volume
    emptyDir: {}
  - name: model-cache-volume
    csi:
      driver: csi.gkm.io
      volumeAttributes:
        csi.gkm.io/GKMCache: llama-3-1-8b-instruct-rocm
        csi.gkm.io/namespace: gkm-test-ns-scoped
```

## Monitoring and Results

### Viewing Logs

```bash
# Watch pod startup and benchmark progress
kubectl logs -f <pod-name> -n gkm-test-ns-scoped

# Get pod status
kubectl get pods -n gkm-test-ns-scoped
kubectl describe pod <pod-name> -n gkm-test-ns-scoped
```

### GPU Resource Verification

```bash
# Check AMD GPU allocation
kubectl describe node | grep -A 5 "amd.com/gpu"

# Verify GPU visibility in pods
kubectl exec -it <pod-name> -n gkm-test-ns-scoped -- rocm-smi
```

## Troubleshooting

### Common Issues

1. **GPU not detected:**
   - Verify ROCm installation: `rocm-smi`
   - Check AMD device plugin deployment
   - Ensure correct device visibility environment variables

2. **Pod fails to start:**
   - Check resource limits and GPU allocation
   - Verify HuggingFace token is correctly configured
   - Review pod logs for specific error messages

3. **Model download issues:**
   - Ensure internet connectivity from pods
   - Verify HuggingFace token has model access permissions
   - Check if model cache is properly configured

4. **Performance issues:**
   - Monitor GPU utilization: `kubectl exec -it <pod-name> -- rocm-smi -d`
   - Adjust vLLM configuration parameters
   - Verify memory settings and limits

### Cleanup

```bash
# Remove all pods
kubectl delete -f k8s/ -n gkm-test-ns-scoped

# Clean up the cluster
./scripts/kubeadm-amd.sh cleanup
```

## References

- [vLLM ROCm Container Documentation](https://rocm.blogs.amd.com/software-tools-optimization/vllm-container/README.html)
- [AMD ROCm Kubernetes Device Plugin](https://github.com/ROCm/k8s-device-plugin)
- [Flannel Networking](https://github.com/flannel-io/flannel)
# Choosing LLM

https://benchlm.ai/agentic

For agentic coding task Qwen/Qwen3.6-27B is well ranked, also it has its own quatization to FP8 https://huggingface.co/Qwen/Qwen3.6-27B-FP8 , esta se puede ver en https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B-FP8 en donde la tabla para MMLUpro

| Configuration | Accuracy | Median Tokens | Tokens/s | Speedup | 
| --- | --- | --- | --- | --- |
| **Qwen3.6-27B base** · standard | 0.902 ± 0.019 | 2186 | 22.69 | 1.00× |
| **Qwen3.6-27B-FP8 (official)** · standard | 0.888 ± 0.007 | 2250 | 35.56 | 1.53× |

which lead us to a recall of 98% (0.888 (base) /0.902 (FP8) ).

Also Qwen3.6-27B-FB8 reaches 90% on SWE-bench https://huggingface.co/Qwen/Qwen3.6-27B/discussions/33

Given this we can use this llm as main model.

# KV Cache — FP8

La ecuación completa es:

$$
\boxed{
KV_{\text{bytes/token}}
=
2
\times
N_{\text{layers}}
\times
N_{\text{KV heads}}
\times
d_{\text{head}}
\times
bytes_{\text{FP8}}
}
$$

Para **Qwen3.6-27B-FP8**:

* $$N_{\text{layers}} = 16$$
* $$N_{\text{KV heads}} = 4$$
* $$d_{\text{head}} = 256$$
* $$bytes_{\text{FP8}} = 1$$
* $$2 = K + V$$

Sustituyendo:

$$
2 \times 16 \times 4 \times 256 \times 1
$$

$$
=32\,768\text{ bytes}
$$

Por tanto:

$$
\boxed{\text{KV Cache}_{FP8}=32\,768\text{ bytes/token}}
$$

o:

$$
\boxed{0.0000305175\text{ GB/token}}
$$

**Resultado: 32 KB de KV cache por token y por secuencia en FP8.**


# Max Concurrent Seqs

Using:

* **HBM:** 80 GB
* **Weights:** 28.7 GB
* **Activations/runtime:** 8 GiB
* **KV bytes/token:** 32 KB = 0.0000305175 GB

Entonces:

$$
\text{max concurrent} =
\frac{80-28.7-8}{0.0000305175 \times \text{max\_len}}
$$

| Max len | KV/request | Concurrent Requests|
| ------: | ---------: | ---------: |
|      8K |    0.24 GB |    **177** |
|     16K |    0.48 GB |     **88** |
|     32K |    0.97 GB |     **44** |
|     64K |       2 GB |     **22** |
|    128K |     3.9 GB |     **11** |
|    256K |     7.8 GB |      **5** |


We will use 64K as max lengh given that in this paper by median https://arxiv.org/pdf/2608.00101 coding task require 68 K tokens per task.

**Hypothesis**: KV memory will be the first limiter at long context lengths; at shorter contexts, compute/scheduling may become the limiter.


# Inference Architecture

## 1. GPU

We will use **2× NVIDIA H100 80 GB**, with one worker per GPU.

The H100 provides enough HBM for the model weights, KV cache, and runtime overhead. A smaller GPU would significantly reduce KV-cache capacity at long context lengths.

## 2. Model

We will use **Qwen3.6-27B-FP8**.

For an FP8 KV cache:

$$
KV_{\text{bytes/token}}
=
2 \times 16 \times 4 \times 256 \times 1
=
\boxed{32,768\text{ bytes/token}}
$$

Therefore:

$$
\boxed{32\text{ KiB/token}}
$$

At **64K tokens**:

$$
32,768 \times 65,536
=
\boxed{2.15\text{ GB per sequence}}
$$

For **22 concurrent requests**:

$$
22 \times 2.15
\approx
\boxed{47.2\text{ GB}}
$$

This fits within our estimated KV-cache budget per H100.

## 3. Topology

We will **not use prefill/decode disaggregation**.

Each worker performs both prefill and decode:

```text
                         Gateway  -------------------
                        /       \                   |
                       /         \                  |
                 H100 #0       H100 #1              |
                  SGLang        SGLang              |
                prefill+decode prefill+decode       |
                    │               │               | 
                local KV         local KV           |
                       \         /                  |
                        \       /                   |
                        Mooncake                    |
                       KV transfer                  |
                                                    │
                                                 overflow
                                                    ▼
                                               Superlinked
```

The **gateway** (which will be litellm) and **engine** are separate components.

* **Gateway:** admission, placement, queueing, and overflow routing.
* **SGLang:** scheduling, RadixAttention, KV-cache management, prefill, and decode.
* **Mooncake:** KV-cache transfer between workers when a useful cached prefix exists on another instance.

## 4. Concurrency

The engine will be configured with:

```text
max_num_seqs  = 22
max_model_len = 64K
```

The 22-request limit is based on:

$$
22 \times 64K \times 32\text{ KiB}
\approx
47.2\text{ GB}
$$

of KV cache per worker.

## 5. KV Transfer

We use **Mooncake** for cross-worker KV-cache transfer.

This is **not** prefill/decode disaggregation. Each worker remains a complete inference instance with its own prefill and decode capacity.

If a useful prefix is cached on another worker, Mooncake can transfer the KV state rather than recomputing the cached prefix.

## 6. Overflow

The **gateway owns admission control**.

If both local SGLang workers are at capacity, the gateway sends the request to **Superlinked** as the overflow backend.

```text
Request
   │
   ▼
Gateway
   │
   ├── local capacity ──► SGLang
   │
   └── local capacity full ──► Superlinked
```

## 7. Scaling

We have **no additional GPU capacity available**, so we will not scale the local GPU pool.

The local capacity is therefore fixed at:

* **2 H100 80 GB**
* **2 SGLang workers**
* **22 concurrent requests per worker**

Additional demand is handled through the **Superlinked overflow path**.

Anyhow if a request can be taken and one worker is at capacity but has the cache for the current request we will transfer the the KV cache to the other replica a keep going.



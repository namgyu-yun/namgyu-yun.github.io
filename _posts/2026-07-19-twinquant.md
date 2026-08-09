---
title: "TwinQuant: Learnable Subspace Decomposition for W4A4 LLM Quantization"
date: 2026-07-19 00:00:00 +0900
categories: [Model Optimization]
tags: [quantization, low-rank, llm-inference]
description: >-
  Notes on TwinQuant (ICML'26), which learns its low-rank/residual
  decomposition directly against quantization error rather than real-valued
  energy, via a fused dual-component kernel.
---

Notes on _TwinQuant_ (ICML'26), a W4A4-INT post-training quantization method for LLMs. Where prior low-rank + quantization approaches (SVD, LoRA-style) minimize the real-valued energy of the residual, TwinQuant instead **learns** the decomposition itself to directly minimize the *quantization error* of both the residual and the low-rank components — via a joint optimization over two learnable transforms, plus a fused dual-component kernel so the extra low-rank branch doesn't cost latency. (ref: [arXiv:2606.01556](https://arxiv.org/pdf/2606.01556))

## 01. Introduction

Prior work decomposes the weight via SVD, then separately trains quantization parameters (scale, clipping threshold) on top. Earlier approaches — _SpinQuant_, _QuaRot_ — apply scaling and (mostly Hadamard) matrix transformations to redistribute weights and activations before quantization, but this incurs high overhead since it requires reading across different coordinate systems.

The more recent _SVDQuant_ (see the [previous post]({% post_url 2026-07-19-svdquant %})) (1) smooths activations first, (2) assumes quantization hardness has moved from activations to weights, and (3) finds the residual $R$ by decomposing the weight into a low-rank outlier component. But SVDQuant (1) incurs high overhead when the rank is large, and (2) leaves a large residual with high accuracy loss when the rank is small.

TwinQuant introduces **learnable subspace decomposition**: it learns how to split each weight into two complementary, quantization-friendly components **under a fixed rank budget**.

> Q. What is the learnable parameter here? How is it computed?
> A. Two matrices, $G$ and $Q$ — covered in §04.

> Q. Can training itself become a source of overhead?
> A. Yes — that's one of the paper's stated limitations (§06).

Note that "learnable" here does **not** mean training quantization parameters like scale or clipping thresholds — TwinQuant doesn't tune those via a full calibration forward pass. It means learning the decomposition itself.

## 02. Background

Integer quantization formula:

$$Q_x = \mathrm{round}\left(\frac{x}{\vec{s}}\right), \quad \vec{s} = \frac{\max(|X|)}{q_{max}}$$

> Note: TwinQuant addresses integer quantization only. Figure 4 (§04) shows INT32 accumulation for Tensor Core MMA.

### A. Low-Rank Decomposition

SVDQuant applies SVD to the scaled weights, absorbing most of the outlier components into a low-rank component while quantizing the remaining residual to low precision:

$$ \hat{Y} = \hat{X}\hat{W} \approx Q(\hat{X})UV + Q(\hat{X})Q(R) $$

<figure>
  <img src="/assets/img/posts/twinquant/spectral-decay.png" alt="Slow singular-value decay in LLM weights" width="600">
  <figcaption>Figure 1. LLM weights exhibit slow spectral decay, leading to rank-sensitive residual quantization error.</figcaption>
</figure>

> Note: **Decay** here means how rapidly singular values shrink when sorted by index. Panel (a) shows *slow* decay — even though the leading singular values drop off quickly, non-negligible ones remain deep into the tail.
>
> If singular values existed only at the front, a handful of leading ranks would capture nearly all the outliers — that's the case for the diffusion-model weights SVDQuant targets, but not for LLM weights.

### B. Parameter-Efficient Learnable Quantization

Slow spectral decay is the crux of the problem TwinQuant sets out to fix (§03), and the fix has to stay parameter-efficient — the whole point of a low-rank branch is a small rank budget, so any learnable component added on top can't reintroduce the cost it was meant to avoid.

## 03. Motivation
### A. Observation 1: LLM Weights Exhibit Slow Singular-Value Decay

See the note under Figure 1 (§02.A).

### B. Observation 2: Slow Spectral Decay Creates a Rank–Overhead Dilemma

Slow decay directly affects residual quantization error. Decomposing into a small rank $r$ leaves large truncated (outlier) values in the residual, so the approximation error grows. Minimizing residual quantization error therefore demands a *large* $r$ to keep more singular values in the low-rank branch — but a large $r$ is exactly the overhead SVDQuant struggles with. This is the dilemma TwinQuant is built to escape.

## 04. Method

<figure>
  <img src="/assets/img/posts/twinquant/framework-overview.png" alt="TwinQuant overall framework">
  <figcaption>Figure 2. The overall framework of TwinQuant.</figcaption>
</figure>

### A. Dual Decomposed Subspaces

TwinQuant reshapes the numerical distribution of the low-rank components, the residual, and the activation, so as to minimize the *quantization* error directly:

$$Error = \hat{Y} - Q(\hat{X})\left[Q(U)Q(V) + Q(R)\right]$$

Assuming quantization noise is mutually independent across terms, this error decomposes into three sources: activation, low-rank weight (outliers), and residual weight (non-outliers). These three are **jointly connected**, so quantizing them under a *fixed* decomposition — as SVDQuant does — can be sub-optimal.

TwinQuant introduces two learnable transforms to address this jointly:

- **Global orthogonal matrices $Q$**: reduce activation and residual error by reshaping their distributions. Being orthogonal, they can be fused into the preceding RMSNorm (as in _QuaRot_), so they add no runtime overhead.
- **Layer-specific invertible matrices $G$**: redistribute the low-rank factors $U, V$. These are invertible rather than orthogonal, trading that structural guarantee for more flexibility.

"Learnable" specifically means: feed calibration data iteratively and compute the reconstruction loss (Eq. 6/9 in the paper) via a forward pass, then update $G$ and $Q$.

<figure>
  <img src="/assets/img/posts/twinquant/rotation-transform.png" alt="Orthogonal and invertible rotation transforms">
  <figcaption>Figure 3. Orthogonal transform (Q) vs. invertible transform (G).</figcaption>
</figure>

> Q. The orthogonal transform should be cheap since it's just a transpose op — but doesn't the invertible transform cost more, since it's a full matmul?
> A. Both $G$ and $Q$ are computed offline, in the calibration stage — so neither adds inference-time overhead.

### B. Joint Optimization on Stiefel and General Linear Manifolds

Since the error involves products of multiple learned matrices ($G$, $Q$), the optimization is non-convex with multiple local minima. Naively optimizing all of them jointly from the start is unstable — so TwinQuant structures the optimization in stages.

#### Hybrid Manifold Optimizer

TwinQuant optimizes with gradient descent — specifically Cayley SGD (to keep $Q$ on its orthogonal/Stiefel manifold) combined with plain momentum SGD for $G$. Nothing exotic beyond respecting each matrix's structural constraint during the update.

#### Stage-Wise Decoupling Training

Jointly optimizing $Q_1, Q_2, G$ re-parameterizes the low-rank factors as:

$$ U' = Q^{-1}UG, \quad V' = G^{-1}V $$

This re-parameterization couples $Q$ and $G$ bilinearly, which makes the joint problem non-convex and hard to optimize directly. To escape that coupling, TwinQuant trains in three stages:

1. **Global Alignment** — optimize $Q_1, Q_2$ only.
2. **Invertible Adaptation** — optimize $G$ only, with $Q$ fixed.
3. **Joint Refinement** — fine-tune $Q$ and $G$ together.

### C. Dual-Component Kernel Design

SVDQuant keeps the low-rank (outlier) branch at high precision. TwinQuant instead quantizes *both* branches — which means the low-rank branch now needs its own efficient kernel path.

<figure>
  <img src="/assets/img/posts/twinquant/dual-component-kernel.png" alt="Fused dual-component low-precision kernel" width="600">
  <figcaption>Figure 4. Workflow of the fused dual-component low-precision kernel.</figcaption>
</figure>

> Q. Why accumulate in INT32 instead of staying 4-bit?
> A. To avoid overflow — a 4-bit×4-bit product (up to 15×15 = 225) already exceeds what 4 bits can represent.

Unlike the residual path, which is a single GEMM ($X \cdot R$), the low-rank path (outliers) is a **two-stage product** ($U \cdot V$). The first GEMM produces an INT32 accumulator — not directly 4-bit-compatible. Naively writing that intermediate result to global memory, then reloading and re-quantizing it for the second GEMM, means extra kernel launches and expensive memory traffic.

#### Fused Two-Stage Low-Rank Pipeline

To avoid that round-trip, TwinQuant re-quantizes the intermediate result **on-chip, in shared memory**, without ever writing it back to global memory. The residual GEMM quantizes activations concurrently, and both paths feed a single shared epilogue.

## 06. Limitations and Future Work

Authors mentioned there are four limitations at _TwinQuant_:

1. Experiments focus on dense models only — no MoE evaluation.
2. **The fused kernel targets the NVIDIA Ada generation (RTX 4090, L20) — no Hopper or Blackwell support.**
3. No validation on long-context datasets.
4. The learnable transform depends on calibration data, and the offline optimization stage takes a long time.

## 07. Conclusion

TwinQuant is a W4A4-INT PTQ workflow built around **learnable subspace decomposition**. A three-stage optimization minimizes quantization error jointly across activation, low-rank, and residual terms, and a fused dual-component kernel keeps the now-quantized low-rank branch from adding latency.

## 08. Personal Comments

_TwinQuant_ is a W4A4-INT (quantize both weight and activation to integer 4-bit) methods for LLM inference. It approaches with **decomposition**, based on unitary-matrix in linear algebra.

Motivated by _SVDQuant_, it uses SVD with more detailed decomposition to identify quantization error. However, since architecture behavior (diffusion vs. transformer-based llm) is different, it is hard to directly apply _SVD_, following _SVDQuant_. Therefore, _TwinQuant_ introduces two learnable (updated within calibration stage) matrix.

Naive implementation would suffer from two-stage issue decreasing throughput (multiple memory read/write), so _TwinQuant_ introduced fused kernel to prevent.

However, it's **experiments are tiny scoped and limitations are left without ablations**.

For example, authors mentioned "no validation on long-context datasets", but didn't experimented with long-context datasets. There is no experiments to check robustness of learnable transform ($Q, G$), because it's experimented within scoped environments.

Plus, since this paper doesn't make oss code, it would be hard to understand $Q, G$ behavior in real production level.

---

## References

- TwinQuant (ICML'26): [arXiv:2606.01556](https://arxiv.org/pdf/2606.01556)
- SVDQuant (ICLR'25 Spotlight): see the [previous post]({% post_url 2026-07-19-svdquant %})
- SpinQuant / QuaRot: rotation-based outlier mitigation for LLM quantization

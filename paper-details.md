This document provides implementation-critical technical specifications for LeJEPA (Latent-Euclidean Joint-Embedding Predictive Architecture) and HunyuanOCR (a specialized Vision-Language Model for OCR).

***

## Technical Summary

| Component | Key Feature | Technical Specification |
| :--- | :--- | :--- |
| **LeJEPA Objective** | Self-Supervised Learning | Minimizes downstream risk by enforcing **Isotropic Gaussian** embedding distribution ($\mathcal{N}(0, I_K)$) via SIGReg. |
| **LeJEPA Complexity** | Efficiency | Linear time and memory complexity $\mathcal{O}(N)$ (minibatch size). Eliminates standard SSL heuristics (e.g., stop-gradient, teacher-student). |
| **HunyuanOCR Arch.** | End-to-End VLM | 1B parameters (0.4B ViT + 0.5B LLM). Uses Native Resolution ViT and XD-RoPE. |
| **HunyuanOCR Training** | Optimization | Four-stage pre-training (454B tokens total) followed by **Group Relative Policy Optimization (GRPO)** reinforcement learning. |

***

## Mathematical Formulations and Equations

### LeJEPA Core Objective

**LeJEPA Total Loss** (Combines prediction loss $\mathcal{L}_{\text{pred}}$ and regularization $\text{SIGReg}$):
$$\mathcal{L}_{\text{LeJEPA}}(\{\mathbf{x}_{n,v}\}_{n,v=1}^{B,V}) = \lambda \frac{1}{V} \sum_{v=1}^V \text{SIGReg}(\{\mathbf{z}_{n,v}\}_{n=1}^B) + \frac{1}{B} \sum_{n=1}^B \mathcal{L}^{(V_g)}_{\text{pred}}(\{\mathbf{z}_{n,v}\}_{v=1}^V)$$

**Prediction Loss ($\mathcal{L}_{\text{pred}}$)**:
$$\mathcal{L}_{\text{pred}} \triangleq \frac{1}{V} \sum_{v'=1}^V \left\| \boldsymbol{\mu}_n - \mathbf{z}_{n,v'} \right\|_2^2$$
where $\boldsymbol{\mu}_n \triangleq \frac{1}{V_g} \sum_{v=1}^{V_g} \mathbf{z}_{n,v}$ (mean of global view embeddings).

**Sketched Isotropic Gaussian Regularization (SIGReg)**:
$$\text{SIGReg}_T(\mathcal{A}, \{ f_{\boldsymbol{\theta}}(\mathbf{x}_n)\}_{n=1}^N) \triangleq \frac{1}{|\mathcal{A}|} \sum_{\mathbf{a} \in \mathcal{A}} T(\{\mathbf{a}^\top f_{\boldsymbol{\theta}}(\mathbf{x}_n)\}_{n=1}^N)$$
The recommended statistical test $T$ is the Epps-Pulley test.

**Epps-Pulley Statistic ($EP$)** (Population level):
$$EP = N \int_{-\infty}^{\infty} \left\| \hat{\phi}_X(t) - \phi(t) \right\|^2 w(t)dt$$
The target characteristic function $\phi(t)$ for $\mathcal{N}(0, 1)$ is $\mathbf{e^{-0.5 t^2}}$.

**Integrated Square Bias (ISB) for k-NN Probing**:
$$\text{ISB}_{k\text{-NN}} = \frac{r_0^4}{(K + 2)^2} \tau_g^2 J(p) + O(r_0^4)$$
The Fisher-information functional $J(p)$ is minimized by the isotropic Gaussian.

### HunyuanOCR Reinforcement Learning

**GRPO Objective Function**:
$$\mathcal{L}_{\text{GRPO}}(\theta) =\mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)}\left[ \frac{1}{G}\sum_{i=1}^G \left[ \min\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \text{clip}\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1 - \epsilon, 1 + \epsilon \right) A_i \right) - \beta D_{\text{KL}} (\pi_{\theta} ||\pi_{\text{ref}}) \right] \right]$$

***

## Algorithms

### LeJEPA SIGReg Implementation (Epps-Pulley Statistic)

```python
def SIGReg(x, global_step, num_slices=256):
    # x is a (N, K) tensor (N=batch size, K=embedding dimension)
    
    # Slice sampling: synchronized across distributed devices
    g = torch.Generator(device=x.device)
    g.manual_seed(global_step)
    proj_shape = (x.size(1), num_slices)
    A = torch.randn(proj_shape, generator=g, device=x.device)
    A /= A.norm(p=2, dim=0) # Normalize directions to unit length

    # Epps-Pulley implementation details
    # Integration points (17 knots recommended)
    t = torch.linspace(-5, 5, 17, device=x.device)
    # Theoretical CF for N(0, 1) and Gaussian window
    exp_f = torch.exp(-0.5 * t**2)

    # Empirical CF (ECF) computation
    x_t = x.unsqueeze(2) * t  # Shape manipulation for projection (N, K) -> (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG") # Aggregate ECF across devices

    # Weighted L2 distance
    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0) * world_size
    
    # Quadrature approximation (Trapezoidal rule)
    T = torch.trapz(err, t, dim=1) * N
    return T
```

### LeJEPA Loss Function

```python
def LeJEPA(global_views, all_views, lambd):
    # global_views: list of global view tensors (length Vg)
    # all_views: list of all view tensors (length V)
    # lambd: scalar trade-off hyperparameter
    
    bs = global_views.size(0) 
    
    # Embeddings (f_theta operation is implicit here as 'forward')
    g_emb = forward(torch.cat(global_views)) # Global view embeddings
    a_emb = forward(torch.cat(all_views))    # All view embeddings

    # LeJEPA loss calculation
    
    # Calculate prediction center (mu_n) from global views
    # g_emb shape: (Vg * bs, K)
    centers = g_emb.view(-1, bs, K).mean(0) # Centers shape: (bs, K)
    
    # Reshape all views for calculation
    # a_emb shape: (V * bs, K)
    a_emb = a_emb.view(-1, bs, K) # Shape: (V, bs, K)
    
    # 1. Prediction loss (sim term = L_pred)
    # (centers broadcast against all views)
    sim = (centers - a_emb).square().mean() 
    
    # 2. SIGReg loss (regularization term)
    # Apply SIGReg independently to each view's embeddings (V times)
    sig_reg = mean(SIGReg(emb, global_step) for emb in a_emb)
    
    # Total LeJEPA Loss
    return (1 - lambd) * sim + lambd * sig_reg
```

***

## Model Architectures and Design Choices

### LeJEPA Architectural Specifications

*   **Encoder ($f_{\boldsymbol{\theta}}$):** Architecture agnostic. Tested architectures include ViTs, ConvNeXts, ResNets, MaxViTs, Swin Transformers.
*   **Target Embedding Distribution:** Isotropic Gaussian, $\mathcal{N}(\mathbf{0}, \mathbf{I})$.
*   **Core Objective:** Maximizing predictive agreement between views and enforcing isotropic Gaussian embedding distribution.
*   **Removed Heuristics (Implementation Critical):**
    *   No stop-gradient.
    *   No teacher–student network architecture.
    *   No predictor network (only $\boldsymbol{\mu}_n$ center used for prediction).
    *   No register tokens required.

| Ablation Parameter | Embedding Dim ($K$) | Projector Dim | Num. Slices ($|A|$) | Frozen Backbone Acc. (%) |
| :--- | :--- | :--- | :--- | :--- |
| ViT-L/14, IN-1K | 512 | 64 | 1024 | 75.29 |
| ViT-L/14, IN-1K | 2048 | 64 | 4096 | 75.65 |
| ViT-L/14, IN-1K | 2048 | 1024 | 4096 | 74.79 |
*Note: The dimensions $K=512, 2048$ and projection dimensions $64, 1024$ were used in ablations, demonstrating robustness across these ranges.*

### HunyuanOCR Architectural Specifications

*   **Total Parameters:** $\mathbf{1\text{B}}$.
    *   Visual Encoder $\approx \mathbf{0.4\text{B}}$ parameters.
    *   Language Model $\approx \mathbf{0.5\text{B}}$ parameters.
*   **Visual Encoder:** **Native Resolution Vision Transformer (Hunyuan-ViT)**.
    *   Base Model: SigLIP-v2-400M pre-trained model.
    *   Input Handling: Supports arbitrary input resolutions via adaptive patching.
    *   Attention: Global attention across all image patches.
    *   Training Strategy: Hybrid generative-discriminative joint training.
*   **Connector:** **Adaptive MLP Connector**.
    *   Functionality: Learnable pooling operation. Performs spatial-dimension adaptive content compression to reduce visual token sequence length.
*   **Language Model:** **Lightweight Hunyuan-0.5B**.
    *   Positional Encoding: **XD-RoPE**.
    *   XD-RoPE Subspaces: **text, height, width, and time** (enables native alignment for 1D text, 2D layout, and 3D spatiotemporal information).

***

## Hyperparameters and Training Procedures

### LeJEPA Training Hyperparameters

| Parameter | Recommended Value(s) | Notes |
| :--- | :--- | :--- |
| Trade-off Hyperparameter ($\lambda$) | $\mathbf{0.05}$ | Stable performance across $\lambda$ values demonstrated. |
| Total Views ($V$) | $\mathbf{8}$ | Configured as $V_g + V_l$. |
| Global Views ($V_g$) | $\mathbf{2}$ | Resolution: $\mathbf{224 \times 224}$. |
| Local Views ($V_l$) | $\mathbf{6}$ | Resolution: $\mathbf{96 \times 96}$. |
| Optimizer | AdamW | Standard implementation. |
| Learning Rate (lr) | $\in \{5\text{e}-3, 5\text{e}-4\}$ | Uses linear warm-up cosine-annealing. |
| Weight Decay (wd) | $\in \{1\text{e}-1, 1\text{e}-2, 1\text{e}-5\}$ | No scheduler on weight decay. |
| Minibatch Size ($N$) | $\mathbf{\ge 128}$ | Achieves competitive performance at $N=128$ on IN-1K. |
| SIGReg Slices ($|A|$) | $\mathbf{1024}$ | Slices are resampled at every step. |
| SIGReg Integration Points | $\mathbf{17}$ | Uses trapezoidal quadrature rule. |
| SIGReg Integration Domain | $\mathbf{[-5, 5]}$ | $t = \text{linspace}(-5, 5, 17)$. |
| Post-training technique (Optional) | Stochastic Weight Averaging (SWA) | Applied to the encoder (to compute $\boldsymbol{\mu}$). |

### HunyuanOCR Pre-Training Recipe (Four Stages)

| Stage | Purpose | Trainable Parts | LR Schedule (Warmup $\to$ Peak/End) | Training Tokens | Sequence Length | Data Composition Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage-1** | V-L Alignment | ViT & Adapter (LM Frozen) | $3\text{e}-4 \to 3\text{e}-5$ | $50\text{B}$ | $\mathbf{8k}$ | General captioning data, synthetic parsing/recognition data, $\le 10\%$ plain text. |
| **Stage-2** | Multimodal Pre-training | All | $2\text{e}-4 \to 5\text{e}-5$ (Warmup-cosine) | $300\text{B}$ | $\mathbf{8k}$ | Synthetic spotting, parsing, translation, VQA data, $\le 10\%$ plain text. |
| **Stage-3** | Long-context Pre-training | All | $8\text{e}-5 \to 5\text{e}-6$ | $80\text{B}$ | $\mathbf{32k}$ | Long pure text, auto-annotated data, long document parsing/IE data. |
| **Stage-4** | Application-oriented SFT | All | $2\text{e}-5 \to 1\text{e}-6$ (Linear decay) | $24\text{B}$ | $\mathbf{32k}$ | Human-annotated data, hard-negative data, standardized instruction data. |

### HunyuanOCR Reinforcement Learning (GRPO) Configuration

| Parameter | Value |
| :--- | :--- |
| Algorithm | **GRPO** |
| Actor Learning rate | $\mathbf{8\text{e}-7}$ (Constant) |
| Optimizer | Adam |
| Micro batch size per GPU | $\mathbf{1}$ |
| Global batch size | $\mathbf{512}$ |
| Max prompt length | $\mathbf{6144}$ |
| Max response length | $\mathbf{16384}$ |
| KL loss coefficient ($\beta$) | $\mathbf{0}$ |
| Rollout Temperature | $\mathbf{0.85}$ |
| N (Responses per prompt) | $\mathbf{8}$ |
| Top-p / Tok-k | $\mathbf{0.95}$ / $\mathbf{50}$ |

***

## Dataset Specifications and Preprocessing Steps

### LeJEPA Data Structure

*   Data format: $(\mathbf{N}, \mathbf{V}, \mathbf{D}) \in \mathbb{N}^{*3}$ tensor.
    *   $N$: Number of independent samples.
    *   $V$: Number of views (augmentations/frames).
    *   $D$: Dimension of each view (e.g., RGB pixels).
*   Data properties required: Samples $\mathbf{x}_n, \mathbf{x}_{n'}$ are independent $\forall n \ne n'$, and identically distributed (i.i.d.).
*   Data Augmentation (Views): Transformations or corruptions (e.g., masking, cropping, blurring, translations, geometric/photometric transformations).
*   Example Datasets: ImageNet-1k, ImageNet-100, Galaxy10, Food101, Flowers102.

### HunyuanOCR Data Pipelines

*   **Total Corpus Size:** Over $\mathbf{200 \text{ million}}$ image-text pairs.
*   **Multilingual Support:** Covers over $\mathbf{130}$ languages.
*   **Image Synthesis Pipeline:** Extension of SynthDog.
    *   Controls: Font, color, orientation, lighting, shadows.
    *   Features: Paragraph-level rendering, bidirectional text (LTR/RTL), complex cursive scripts.
*   **Image Augmentation Pipeline:** In-house **Warping Synthesis Pipeline**.
    *   Geometric Deformation: Control-point manipulation to emulate folds, curves, perspective distortions.
    *   Imaging Degradation: Motion blur, Gaussian noise, compression artifacts.
    *   Illumination Perturbations: Global/local lighting variations, shadows, reflections.
*   **QA Generation Pipeline:** Automated generation from spotting/parsing outputs.
    *   Hard Sample Retrieval: Filters for low clarity, complex tables/formulas, code snippets, low-resource languages.
    *   Consistency Verification: Multi-model cross-validation to assess confidence of generated QA pairs.

### HunyuanOCR Instruction Templates (Prompting)

| Task Type | Output Format Specification | Example Prompt (English) |
| :--- | :--- | :--- |
| **Spotting** | `<ref>text</ref><quad>(x1,y1),(x2,y2)</quad>`. Coordinates normalized to $\mathbf{}$. | "Detect and recognize text in the image, and output the text coordinates in a formatted manner." |
| **Formula Parsing** | **LATEX** format. | "Identify the formula in the image and represent it using LATEX format." |
| **Table Parsing** | **HTML** format. | "Parse the table in the image into HTML." |
| **Chart Parsing** | **Mermaid** (flowcharts) or **Markdown** (other charts) format. | "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts." |
| **Full Document Parsing** | **Markdown** for text, **HTML** for tables, **LATEX** for formulas, reading order preserved. | "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LATEX format, and the parsing should be organized according to the reading order." |
| **IE (Multi-field)** | **JSON** format based on key list. | "Extract the content of the fields: [‘key1’,‘key2’, . . . ] from the image and return it in JSON format." |

***

## Evaluation Metrics and Benchmarks

### LeJEPA Evaluation Metrics

*   **Metric:** Top-1 accuracy (%).
*   **Method:** Linear evaluation with frozen backbone (standard).
*   **Training Loss Quality Metric:** Spearman correlation ($\rho_s$) between $\mathcal{L}_{\text{LeJEPA}}$ and downstream accuracy.
    *   Scaled Loss Correlation: $\mathcal{C}(\alpha) = \rho_s \left( \frac{\text{train\_loss}}{\lambda^\alpha}, \text{test\_accuracy} \right)$.
    *   Optimal Scaling Coefficient: $\mathbf{\alpha \approx 0.4}$ achieves up to $99\%$ correlation.

| Model | Pretrain Data | Epochs | Params | Metric | Result (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LeJEPA ViT-H/14 | ImageNet-1k | N/A | N/A | Linear Probe Acc | $\mathbf{79.0}$ |
| LeJEPA ConvNeXtV2-H | IN-1K | 100 | 660M | Linear Probe Acc | $\mathbf{78.5}$ |
| LeJEPA ConvNeXt-V2 Nano | Galaxy10 | N/A | N/A | Full Finetuning | $\mathbf{82.72}$ |
| DINOv3 ViT-S/16 | Transfer (Natural Images) | N/A | N/A | Full Finetuning | $81.60$ |
| LeJEPA ViT-L | IN-1K | 100 | 304M | Few-shot Avg (10-shot) | $\mathbf{60.95}$ |
| I-JEPA ViT-H | IN-1K | 300 | 632M | Few-shot Avg (10-shot) | $60.51$ |

### HunyuanOCR Evaluation Metrics

*   **Spotting:** Overall accuracy on 9-category in-house benchmark (900 images).
*   **Parsing:** OmniDocBench/Wild-OmniDocBench official protocol. DocML uses overall edit-distance–based score.
*   **IE/VQA:** Exact-match accuracy (for structured JSON outputs). OCRBench standard protocol.
*   **Translation:** **COMET** score.

| Task | Benchmark | HunyuanOCR (1B Param) Score | Closest SOTA Performance | SOTA Model (Param Size) |
| :--- | :--- | :--- | :--- | :--- |
| **Parsing** | OmniDocBench (Overall $\uparrow$) | $\mathbf{94.10}$ | $91.93$ | PaddleOCR-VL (0.9B) |
| **IE/VQA** | Cards/Receipts (Exact Match $\uparrow$) | $\mathbf{92.29}$ / $\mathbf{92.53}$ | $80.59$ / $80.66$ | Gemini-2.5-Pro (N/A) |
| **VQA** | OCRBench | $860$ | $920$ | Qwen3-VL-235B (235B) |
| **Translation** | DoTA (en2zh COMET $\uparrow$) | $\mathbf{83.48}$ | $82.09$ | PP-DocTranslation (N/A) |

***

## Implementation-Critical Design Choices

### LeJEPA Stability and Complexity

*   **Complexity:** Linear time and memory complexity $\mathcal{O}(N)$ (minibatch size).
*   **Gradient Stability:** Epps-Pulley test guarantees uniformly bounded loss, gradient, and curvature regardless of input distribution.
    *   Gradient norm bound: $\left\| \nabla_{\theta} EP(\mathbf{a}) \right\| \le \frac{4\sigma^2}{N} \sum_{i=1}^N \left\| \mathbf{a}^\top \nabla_{\theta} f_{\theta}(\mathbf{x}_i) \right\|$.
*   **Dimensionality Robustness:** SIGReg defeats the curse of dimensionality by leveraging the smoothness ($\alpha$) of the embedding density. Error decay rate is bounded by $O(|A|^{-2\alpha/(K-1)})$ for $|A|$ slices.
*   **DDP Compatibility:** SIGReg is DDP-friendly and scalable via efficient `all_reduce` operations for computing the Empirical Characteristic Function (ECF).

### HunyuanOCR Training Strategy

*   **Optimization Paradigm:** Reinforcement Learning with Verifiable Rewards (RLVR) for closed-form tasks and LLM-as-a-judge approach for open-ended tasks.
*   **RL Reward Definition (Spotting):** Maximize IoU between predicted and ground-truth bounding boxes AND minimize normalized edit distance between recognized text.
*   **RL Reward Definition (Translation):** Soft reward score $\in$ normalized to $\mathbf{}$, designed to expand reward granularity in the $\mathbf{2 \text{ to } 4}$ mid-range.
*   **Context Window Extension:** Stage-3 pre-training implements long-context capability up to $\mathbf{32k}$ sequence length.
*   **Positional Encoding:** **XD-RoPE** is mandatory for leveraging 2D layout and 3D spatiotemporal information within the LLM component.
This implementation plan outlines the execution strategy for **"Hunyuan-LeJEPA"**—a hybrid OCR model that replaces the standard supervised/contrastive vision backbone of HunyuanOCR with a **LeJEPA-pretrained encoder**.

This combination leverages LeJEPA’s data efficiency and stability to create a visual encoder that understands document structure (layout, reading order) from unlabeled data, before being aligned with an LLM for text transcription.

### **Phase 1: High-Efficiency Data Strategy (Revised)**
**Goal:** Leverage LeJEPA's efficiency to train on **2M high-quality documents** (1% of the original plan), rather than 200M.

#### **1.1. The "Smart" Mixture (2M Total)**
Instead of scraping the web blindly, we construct a balanced dataset from open-source repositories.

* **50% Layout Diversity (1M images):**
    * **Source:** **IIT-CDIP Test Collection** (Library of Congress).
    * *Why:* Contains older scans, letters, memos, and forms. High noise, perfect for learning robustness.
    * *Action:* Randomly sample 1M pages.
* **30% Digital Born PDFs (600k images):**
    * **Source:** **Common Crawl (PDF subset)** or Hugging Face `pixparse/pdfa-eng-wds`.
    * *Why:* Clean digital text, complex layouts, modern fonts.
* **20% Dense Information (400k images):**
    * **Source:** **Cord** (Consolidated Receipt Dataset) and **DocVQA**.
    * *Why:* Receipts, invoices, and tables. Forces the model to look at fine-grained spatial relationships.

**Where to get it:**
* Hugging Face Datasets: `rvl_cdip`, `naver-clova-ix/cord-v2`, `pixparse/pdfa-eng-wds`.

---

### **Phase 2: Visual Pre-Training (LeJEPA)**
**Goal:** Train the Hunyuan-ViT to understand document structure without labels.

#### **2.1. The Architecture**
* **Encoder:** Hunyuan-ViT (Frozen configuration initially).
* **Projector (The LeJEPA Head):** 3-Layer MLP. Output dimension = 8192.
    * *Note:* We use a wide output (8192) to force the features to spread out (Gaussian expansion).

#### **2.2. Hyperparameters (LeJEPA Specific)**
* **Batch Size:** 2048 (Use gradient accumulation if VRAM is low). LeJEPA needs reasonable batch sizes for the statistical tests.
* **Learning Rate:** $1e^{-4}$ (lower than standard contrastive learning).
* **Weight Decay:** $1e^{-6}$ (LeJEPA prefers low weight decay to allow feature expansion).
* **Masking Strategy (Crucial):**
    * Do *not* use random patch masking (like MAE).
    * Use **Block-wise Masking:** Mask out 30-50% of the image in large contiguous blocks (e.g., mask the whole "Total" section of a receipt). This forces the model to predict *context*, not just edge reconstruction.

---

### **Phase 3: The Transition (Fixing the Shift)**
**Goal:** Map the LeJEPA features to the LLM *without* destroying the learned geometry.

#### **3.1. The "Adapter Bridge" Strategy**
We cannot simply discard the projector. The features *before* the projector are "Pre-Gaussian." The features *after* are "Gaussian."
Since the LLM expects specific token dimensions, we cannot keep the 8192-dim LeJEPA Projector.

**The Solution: Two-Stage Adapter Alignment.**

**Step A: The "Probe" Initialization (Frozen ViT)**
1.  **State:** Freeze the ViT completely. Discard the LeJEPA Projector. Initialize the **Hunyuan Adapter** (Adaptive MLP).
2.  **Constraint:** The ViT output is currently *misaligned* with the LLM.
3.  **Training:** Train *only* the Adapter.
4.  **Learning Rate:** High ($1e^{-3}$).
5.  **Data:** Use **Text-Image Pairs** (e.g., 500k captioned images).
6.  **Why:** This treats the ViT features as a fixed "Gold Standard." The Adapter learns to translate "LeJEPA dialect" into "LLM dialect." **Do not unfreeze the ViT yet.**

**Step B: Gentle Unfreezing (The Handshake)**
1.  **State:** Unfreeze the ViT, but set a **differential learning rate**.
2.  **ViT LR:** Very low ($1e^{-6}$). We want to *nudge* the ViT, not shatter its weights.
3.  **Adapter/LLM LR:** Standard ($1e^{-5}$).
4.  **Why:** This allows the ViT to slightly adjust its features to be more "readable" for the LLM without losing the robust structure learned in Phase 2.

---

### **Phase 4: Execution Checklist**

#### **Step 1: Environment & Data (Days 1-2)**
* [ ] Download `rvl_cdip` and `cord` from Hugging Face.
* [ ] Pre-process: Resize to max 1024x1024 (maintain aspect ratio, pad with gray).
* [ ] Create "Global" (full image) and "Local" (crop) views loader.

#### **Step 2: LeJEPA Pre-training (Days 3-5)**
* [ ] **Hardware:** 4x A100 (or 8x A10G/4090s).
* [ ] **Loss Monitoring:** Watch the **SIGReg Loss**. It must converge to a low constant value. If it spikes, your embeddings are collapsing—increase the `lambda` (regularization strength).
* [ ] **Checkpoint:** Save the model when the "Prediction Loss" plateaus.

#### **Step 3: Alignment (Days 6-7)**
* [ ] **Switch Architecture:** Load ViT weights. Drop Projector. Add Adapter + LLM.
* [ ] **Data:** Switch to `OCR-VQA` or synthetic LaTeX renderings (text-pair data).
* [ ] **Run Step A (Frozen):** 1 epoch.
* [ ] **Run Step B (Unfrozen):** 2 epochs.

### **Summary of Corrections**
| Feature | Old Plan (Flawed) | **New Plan (Corrected)** |
| :--- | :--- | :--- |
| **Data Scale** | 200M Images | **2M Curated Images** (LeJEPA efficiency). |
| **Transition** | Discard Projector $\to$ Train | **Freeze ViT $\to$ High-LR Adapter Probe $\to$ Gentle Unfreeze.** |
| **Projector** | Discarded entirely | Used to shape ViT, then discarded *only after* ViT weights are locked. |
| **Data Source** | "Web Crawl" | **Hugging Face (`rvl_cdip`, `cord`, `pdfa`).** |
# MELD Multimodal Emotion Recognition in Conversations: Architecture Proposals Toward SOTA (≥70% Weighted-F1)

## Table of Contents

1. [Dataset & Task Definition](#1-dataset--task-definition)
2. [Current SOTA Landscape](#2-current-sota-landscape)
3. [Computational Constraints & Training Environments](#3-computational-constraints--training-environments)
4. [Task Decomposition: Nine Independent + One Fusion Task](#4-task-decomposition)
5. [Task 1 — Text-Only Emotion Recognition](#task-1--text-only-emotion-recognition)
6. [Task 2 — Audio-Only Emotion Recognition](#task-2--audio-only-emotion-recognition)
7. [Task 3 — Video-Only (Facial + Scene) Emotion Recognition](#task-3--video-only-facial--scene-emotion-recognition)
8. [Task 4 — Pose-Only Emotion Recognition](#task-4--pose-only-emotion-recognition)
9. [Task 5 — Text + Audio (Speech-Text) Emotion Recognition](#task-5--text--audio-speech-text-emotion-recognition)
10. [Task 6 — Text + Video Emotion Recognition](#task-6--text--video-emotion-recognition)
11. [Task 7 — Audio + Video Emotion Recognition](#task-7--audio--video-emotion-recognition)
12. [Task 8 — Text + Audio + Video Emotion Recognition](#task-8--text--audio--video-emotion-recognition)
13. [Task 9 — Full Multimodal Fusion (Text + Audio + Video + Pose + Speaker + Context)](#task-9--full-multimodal-fusion)
14. [Class Imbalance: Strategies Per Task](#14-class-imbalance-strategies-per-task)
15. [Modular Hyperparameter Protocol](#15-modular-hyperparameter-protocol)
16. [Feature Extraction Pipeline & Storage Strategy](#16-feature-extraction-pipeline--storage-strategy)
17. [Evaluation Protocol](#17-evaluation-protocol)
18. [Recommended Libraries & Dependencies](#18-recommended-libraries--dependencies)
19. [References](#19-references)

---

## 1. Dataset & Task Definition

### 1.1 MELD CSV Schema

Each row in `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv` contains:

|Column|Type|Description|
|---|---|---|
|`Sr No.`|int|Global utterance index|
|`Utterance`|str|Raw transcript text|
|`Speaker`|str|Character name (e.g., "Ross", "Monica")|
|`Emotion`|str|**Primary label** — one of 7 classes|
|`Sentiment`|str|Coarser label — positive / negative / neutral|
|`Dialogue_ID`|int|Conversation index|
|`Utterance_ID`|int|Position within dialogue|
|`Season`|int|TV season|
|`Episode`|int|TV episode|
|`StartTime`|str|Clip start timestamp in episode|
|`EndTime`|str|Clip end timestamp|

**Media files**: `dia{Dialogue_ID}_utt{Utterance_ID}.mp4` (video+audio) and `.wav` (audio-only) in `train_splits/`, `dev_splits/`, `test_splits/`.

### 1.2 Emotion Classes & Distribution

|Emotion|Train|Dev|Test|% of Train|
|---|---|---|---|---|
|neutral|4,710|470|1,256|~47%|
|surprise|1,205|150|281|~12%|
|fear|268|40|50|~2.7%|
|sadness|683|111|208|~6.8%|
|joy|1,743|163|402|~17%|
|disgust|268|22|68|~2.7%|
|anger|1,109|153|345|~11%|

**Critical observation**: `neutral` dominates at ~47%. `fear` and `disgust` are severely underrepresented (~2.7% each). Any architecture failing to address this imbalance will collapse toward predicting neutral and report inflated macro accuracy while underperforming on weighted-F1.

### 1.3 Mathematical Task Formulation

A conversation $C = {u_1, u_2, \ldots, u_N}$ consists of $N$ utterances. Each utterance $u_i$ is spoken by speaker $s_i \in S$ and has associated modalities:

$$u_i = (x_i^T, x_i^A, x_i^V, x_i^P, s_i, \text{ctx}_i)$$

where $x_i^T$ = text, $x_i^A$ = audio, $x_i^V$ = video frames, $x_i^P$ = pose, and $\text{ctx}_i = {u_{i-w}, \ldots, u_{i-1}}$ is the dialogue history window of size $w$.

The ERC objective is to learn $f: u_i \mapsto y_i \in {\text{anger, disgust, fear, joy, neutral, sadness, surprise}}$.

The evaluation metric is **Weighted F1**:

$$\text{WF1} = \sum_{c=1}^{7} \frac{|C_c|}{N} \cdot F1_c, \quad F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

where $|C_c|$ is the count of samples in class $c$.

---

## 2. Current SOTA Landscape

Understanding the ceiling helps set realistic targets.

### 2.1 Text-Only SOTA on MELD

|Model|WF1 (MELD)|Notes|
|---|---|---|
|InstructERC (LLaMA2-7B + LoRA)|69.15|Retrieval-augmented, multi-task|
|BiosERC (LLaMA2-13B + LoRA)|69.83|Speaker biography injection|
|LaERC-S (LLaMA2-7B + LoRA)|69.27|Speaker characteristics via oReact|
|PRC-Emo (Qwen3-8B + LoRA)|70.44|Prompt + Retrieval + Curriculum (2025)|
|MiSTER-E (MoE speech+text)|67.9|Text+Audio only|

**Target for Task 1 (text-only): ≥70.44 WF1** — matching or exceeding PRC-Emo.

### 2.2 Multimodal SOTA on MELD

|Model|Modality|WF1|Notes|
|---|---|---|---|
|MaTAV|T+A+V|~66.9 (Acc)|Mamba-based alignment|
|SALM|T+A+V|67.13|Liquid-Mamba, optimal transport alignment|
|DER-GCN|T+A+V|66.10|Dialog+event relation GCN|
|MMGCN|T+A+V|58.65|GCN-based fusion|
|Quality-Ctrl+Mamba|T+A+V|64.3 WF1|Identity-transfer learning|

**Target for Task 9 (full fusion): ≥70 WF1** — surpassing all published multimodal results.

### 2.3 Key Insight from Literature

The text modality alone at ~69–70% WF1 often **outperforms full multimodal systems** on MELD. This is because:

1. MELD audio/video is extracted from TV broadcasts with TV production noise
2. Facial expressions are often occluded or off-camera
3. Multiple speakers appear in frames simultaneously
4. Audio has background music and audience laughter

This motivates a **late-fusion strategy** where each modality is independently strong before fusion, rather than early fusion relying on poor visual/audio features dragging text performance down.

---

## 3. Computational Constraints & Training Environments

### 3.1 Hardware Profiles

|Environment|GPU|VRAM|RAM|Use Case|
|---|---|---|---|---|
|Local (1650 SUPER)|NVIDIA GTX 1650 SUPER|4 GB|32 GB|Feature pre-extraction only|
|Colab Free|T4 / A100 (limited)|15 GB / 40 GB|12 GB|Light fine-tuning, Task 1 text|
|Kaggle Free|P100 / T4 (30 hr/week)|16 GB|13 GB RAM|Main training environment|

### 3.2 Memory Budgets Per Task

Feature extraction is decoupled from training — all raw features are pre-extracted and saved as `.npy` or `.pt` tensors. This avoids reprocessing video/audio during each training epoch.

**Recommended workflow:**

1. Extract all modality features offline → save to disk per utterance
2. Training loads pre-computed features → far faster iteration
3. No GPU needed for inference on text features (CPU-viable for BERT/RoBERTa base)

### 3.3 Mixed Precision & Gradient Checkpointing

All tasks must use `torch.cuda.amp` (automatic mixed precision, fp16) and gradient checkpointing where applicable (critical for LLM fine-tuning). This halves VRAM requirements.

---

## 4. Task Decomposition

Each of the 9 tasks below is **independently trainable** with its own feature set, model, loss, and hyperparameters. Task 9 is a late/intermediate fusion of Tasks 1–8 outputs. This modularity allows:

- Swapping a backbone in Task 2 without affecting Task 3
- Ablating which modality contributes most
- Identifying the performance ceiling per modality
- Combining the strongest unimodal and bimodal configurations for Task 9

### 4.1 Task Index

|Task|Input Modalities|Primary Goal|
|---|---|---|
|1|Text (utterance + context)|≥70.44 WF1|
|2|Audio (WAV features)|≥55 WF1|
|3|Video (face + scene)|≥50 WF1|
|4|Pose (body keypoints)|≥45 WF1|
|5|Text + Audio|≥71 WF1|
|6|Text + Video|≥70 WF1|
|7|Audio + Video|≥60 WF1|
|8|Text + Audio + Video|≥72 WF1|
|9|All modalities + Speaker + Context|≥73 WF1|

---

## Task 1 — Text-Only Emotion Recognition

### Overview

The text modality is the single strongest signal in MELD. A carefully fine-tuned LLM with conversation context, speaker characteristics, and curriculum learning can achieve ≥70% WF1.

### Architecture A: Discriminative RoBERTa-Large + Context Window (Baseline, Fast)

**Input representation:** Given utterance $u_i$ and context window $w = 5$, construct the input string:

```
[CLS] {speaker_{i-4}}: {utt_{i-4}} [SEP] ... {speaker_i}: {utt_i} [SEP]
```

The `[CLS]` embedding serves as the utterance representation. This is the same input construction as in CoMPM and BiosERC's BERT-based variant.

**Mathematically:**

$$h_i = \text{RoBERTa}([\text{CLS}; u_{i-w}; \ldots; u_i])[\text{CLS}]$$

$$\hat{y}_i = \text{Softmax}(W_c h_i + b_c), \quad W_c \in \mathbb{R}^{d \times 7}$$

**Key design choices:**

- Use `roberta-large` (355M params, d=1024) over `roberta-base` for +1–2% WF1
- Context window $w \in {3, 5, 7}$ — tune on dev set
- Include speaker names in the text prefix (not just anonymized Speaker_0/1)
- Speaker names in MELD are meaningful (Ross, Monica, etc. have known personalities to pretrained models)

**Loss:** Weighted cross-entropy:

$$\mathcal{L} = -\sum_{c=1}^7 w_c \cdot y_c \log \hat{y}_c$$

where $w_c = \frac{N}{7 \cdot N_c}$ (inverse frequency weighting). This is critical for fear/disgust.

**Libraries:** `transformers` (HuggingFace), `torch`, `sklearn`

**Expected WF1:** 65–67%

---

### Architecture B: LLM Fine-tuning with LoRA (Primary, Targets ≥70%)

This architecture follows the InstructERC → BiosERC → PRC-Emo lineage but adds key improvements.

**Base model options (all runnable on Kaggle P100/T4 16GB VRAM with LoRA):**

|Model|Size|VRAM (LoRA, r=16)|Notes|
|---|---|---|---|
|LLaMA2-7B|7B|~8 GB fp16|Baseline, proven on MELD|
|Mistral-7B-v0.3|7B|~8 GB fp16|Often better than LLaMA2-7B|
|Qwen2.5-7B-Instruct|7B|~8 GB fp16|PRC-Emo used this for IEMOCAP|
|Qwen3-8B|8B|~10 GB fp16|PRC-Emo MELD best (70.44%)|

**LoRA configuration:**

Low-Rank Adaptation (Hu et al., 2022) adds trainable matrices $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ to frozen weight $W_0$:

$$W = W_0 + \Delta W = W_0 + BA, \quad \text{with } r \ll d$$

The update $\Delta W = BA$ has rank $r$, greatly reducing trainable parameters. For a 7B model, LoRA with $r=16$, $\alpha=32$ reduces training parameters to ~0.1% of total.

**PEFT (Parameter-Efficient Fine-Tuning) targets:** Apply LoRA to all attention projections: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

**Instruction template (adapted from PRC-Emo + LaERC-S):**

```
<system>
You are an expert in analyzing emotions in conversations.
Given the conversation history, speaker characteristics, 
and emotion interpretations, predict the emotion label.
</system>

<context>
{dialogue_history: last w utterances with speaker names}
</context>

<speaker_info>
{speaker_name} characteristics: {LLM-generated biography ~100 words}
</speaker_info>

<emotion_interpretation>
Explicit emotion of {speaker_name} in "{current_utterance}": {explicit_interp}
Implicit emotion of {speaker_name} in "{current_utterance}": {implicit_interp}
</emotion_interpretation>

<demonstrations>
{top-3 similar utterances from retrieval repository with gold labels}
</demonstrations>

<question>
Based on all the above, what is the emotion of {speaker_name} 
in "{current_utterance}"?
Available labels: anger, disgust, fear, joy, neutral, sadness, surprise
</question>

<answer>
{emotion_label}
</answer>
```

**Template components breakdown:**

|Component|Input|Output|Dependency|
|---|---|---|---|
|`dialogue_history`|CSV `Utterance`, `Speaker`, window context|Formatted string|pandas, context window $w$|
|`speaker_info`|Full conversation|~100-word LLM description|Pre-extracted offline with Qwen3-14B|
|`explicit_interp`|Historical utterances|20–50 word explicit emotion|Pre-extracted offline|
|`implicit_interp`|Historical utterances|20–50 word implicit emotion|Pre-extracted offline|
|`demonstrations`|Current utterance text|Top-3 (utterance, label) pairs|SBERT retrieval repository|
|`answer`|—|Single emotion label|Ground truth during training|

**Offline pre-extraction (done once before training):**

Step 1: Use a stronger LLM (Qwen3-14B, free via Groq API or local 8-bit quantization) to generate:

- Speaker biography per dialogue
- Explicit + implicit emotion interpretations per utterance

Step 2: Store all generated texts as CSV columns: `speaker_bio`, `explicit_emotion`, `implicit_emotion`

Step 3: Build SBERT retrieval repository:

- Encode all training utterances with `sentence-transformers/all-mpnet-base-v2`
- Index with FAISS (`IndexFlatIP` for cosine similarity)
- At training time, retrieve top-3 nearest neighbors (excluding itself)

**Curriculum Learning:**

Following PRC-Emo (Li et al., 2025), define a dialogue difficulty score:

$$\text{DIF}(c_i) = \frac{W!E!S_{\text{same}}(c_i) + W!E!S_{\text{diff}}(c_i) + N_{sp}(c_i)}{N_u(c_i) + N_{sp}(c_i)}$$

where:

$$W!E!S = \sum_j (k \cdot s_{ij} + b)$$

$s_{ij}$ = emotional similarity between consecutive emotions (via valence-arousal wheel cosine similarity), $k$ = 0.8, $b$ = 0.2 (tunable). $N_{sp}$ = number of unique speakers, $N_u$ = total utterances.

**Training schedule:**

```
Epoch 1–2: Train on bottom-50% difficulty bucket only
Epoch 3–4: Add top-50% difficulty (full dataset)
Total epochs: 4 (Kaggle T4 can fit ~3–4 epochs for 7B LoRA in 30hr session)
```

**Loss function:** Focal loss with class weighting:

$$\mathcal{L}_{\text{focal}} = -\sum_c w_c (1 - \hat{p}_c)^\gamma y_c \log \hat{p}_c$$

with $\gamma = 2.0$, $w_c = \frac{1}{\sqrt{N_c}}$ (square-root damping, less aggressive than full inverse).

**Training hyperparameters (Kaggle T4/P100):**

|Hyperparameter|Value|Source|
|---|---|---|
|LoRA rank $r$|16–32|PRC-Emo uses 32|
|LoRA $\alpha$|$2r$|Standard|
|LoRA dropout|0.05||
|Learning rate|3e-4|PRC-Emo|
|Batch size|1 (+ grad accumulation 4)|Memory constraint|
|Gradient accum. steps|4–8|Effective batch = 4–8|
|Warmup ratio|0.03||
|Max sequence length|1024–2048 tokens||
|Epochs|4|PRC-Emo|
|Optimizer|AdamW||
|LR scheduler|Linear decay||
|bf16|True (if A100) / fp16 (T4)||
|Gradient checkpointing|True|Saves VRAM|

**Expected WF1:** ≥70.44 (matching PRC-Emo, potentially exceeding with better class balancing)

---

### Architecture C: Contrastive + Discriminative BERT (Alternative, Lighter)

For faster iteration, use RoBERTa-Large with supervised contrastive loss:

$$\mathcal{L}_{\text{SupCon}} = -\sum_{i} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} \exp(z_i \cdot z_a / \tau)}$$

where $z_i = \text{normalize}(h_i)$, $P(i)$ = set of same-class samples in batch, $\tau = 0.07$.

This is combined with cross-entropy in a multi-task fashion:

$$\mathcal{L} = \lambda \mathcal{L}_{\text{CE}} + (1-\lambda) \mathcal{L}_{\text{SupCon}}, \quad \lambda \in [0.3, 0.7]$$

**Why contrastive helps**: MELD's class imbalance means representations for fear/disgust are rarely seen. SupCon explicitly pulls together same-emotion embeddings regardless of frequency, creating denser clusters for rare classes.

---

## Task 2 — Audio-Only Emotion Recognition

### Context

MELD audio is notoriously noisy (TV broadcast, background music, audience). Pure audio models typically achieve 55–62% WF1 on MELD. The goal here is ≥55%, with upside to ~62%.

### Feature Extraction Options

All features are extracted offline and stored as `.npy` per utterance.

**Option A: WavLM-Large (Recommended)**

WavLM (Chen et al., 2022) is a self-supervised speech model trained on 94,000 hours of speech with masked speech prediction and denoising. It outperforms wav2vec2 on most downstream tasks.

Process:

1. Load `.wav` file
2. Resample to 16,000 Hz (WavLM's expected rate)
3. Pad/truncate to fixed length (e.g., 5 seconds = 80,000 samples)
4. Extract from `WavLMModel` — take weighted sum of all transformer layers (trainable scalar weights $\alpha_l$):

$$h^A_i = \sum_{l=1}^{L} \alpha_l \cdot H_i^l$$

where $H_i^l$ is the $l$-th layer hidden state, averaged over time dimension → shape $(d_{wavlm},)$ = (1024,) for WavLM-Large.

**Library:** `transformers` — `WavLMModel`, `AutoFeatureExtractor`  
**VRAM for extraction:** ~4 GB (WavLM-Large is 316M params, runs on 1650 SUPER for feature extraction)

**Option B: Wav2Vec2-XLSR-53 (Lighter)**

Pre-trained on 53 languages. Lower capacity than WavLM-Large but faster.

**Option C: Emotion2Vec (Emotion-specific pretraining)**

`emotion2vec` (Ma et al., 2023) is specifically pre-trained for speech emotion using self-supervised learning on speech emotion datasets. Often outperforms general speech models on downstream emotion tasks.

- Available: `iic/emotion2vec_plus_large` on HuggingFace / ModelScope
- Extracts 768-d or 1024-d embeddings directly emotion-tuned

**Option D: OpenSMILE handcrafted features (LLD + HSF)**

Extract Low-Level Descriptors (LLDs): pitch, energy, MFCCs (39 coefficients), spectral features, voice quality (jitter, shimmer). Apply functionals (mean, std, min, max, percentiles) over utterance → fixed-length vector ~384-d.

**Library:** `opensmile` Python package — use ComParE or IS09 feature set  
**When to use**: Ensemble with deep features, or as fallback when GPU unavailable

### Architecture: Conformer-based Utterance Classifier

**Input:** Pre-extracted WavLM-Large features of shape $(T, 1024)$ where $T$ = number of frames

**Step 1 — Temporal modeling with Conformer**

Conformer (Gulati et al., 2020) combines CNNs (local patterns) and Transformers (global context), highly effective for speech:

$$\text{Conformer}(h) = \text{LayerNorm}(\text{MHSA}(\text{Conv}(h) + h) + h)$$

More precisely, each Conformer block has:

1. Feed-Forward (FF) module (half-step)
2. Multi-Head Self-Attention (MHSA)
3. Depthwise Separable Convolution module
4. Feed-Forward module (half-step)
5. LayerNorm

Use 4–8 Conformer blocks, hidden dim 256–512.

**Step 2 — Utterance pooling**

$$h^A_{\text{utt}} = \text{AttentivePool}(H) = \sum_t \alpha_t h_t, \quad \alpha_t = \text{softmax}(w^T \tanh(W h_t))$$

This learned attentive pooling is better than mean/max pooling for speech — it focuses on emotionally salient frames.

**Step 3 — Classification head**

$$\hat{y} = \text{Softmax}(\text{Dropout}(W_2 \text{GELU}(W_1 h^A_{\text{utt}})))$$

**Loss:** Weighted focal loss (same as Task 1) — critical for fear/disgust.

**Modular swap points:**

- Temporal module: Conformer → BiLSTM → Mamba2 → Temporal Transformer
- Pooling: Attentive → CLS token → Mean → Max
- Input features: WavLM-Large → emotion2vec → OpenSMILE

**Training hyperparameters:**

|Hyperparameter|Value|
|---|---|
|Conformer layers|4–6|
|Hidden dim|256|
|Attention heads|4|
|Kernel size (conv module)|31|
|Dropout|0.1–0.2|
|Learning rate|1e-4|
|Batch size|32|
|Epochs|30–50|
|Optimizer|AdamW + cosine LR decay|
|Focal $\gamma$|2.0|

### Extended Audio: Context-Aware Audio Model

To model dialogue context in audio, stack audio features across the context window:

Given ${h^A_{i-w}, \ldots, h^A_i}$, apply a Transformer encoder:

$$H^A_{\text{ctx}} = \text{TransformerEncoder}([h^A_{i-w}; \ldots; h^A_i])$$

Target: $h^A_i$ position output for classification.

This is inspired by CoMPM's context modeling for text, adapted to audio. Expected +2–3% WF1.

---

## Task 3 — Video-Only (Facial + Scene) Emotion Recognition

### Context

Video emotion recognition on MELD is the hardest single modality due to:

- Multiple people in frame
- Facial occlusion, lighting changes
- Fast cuts between scenes
- Emotional expression in TV often exaggerated or context-dependent

Expected ceiling: ~50–55% WF1. This modality contributes in fusion but rarely competes with text alone.

### Feature Extraction Pipeline

**Step 1 — Frame extraction**

From each `.mp4` clip, extract $K$ uniformly sampled frames (e.g., $K = 8$):

```
ffmpeg -i dia{d}_utt{u}.mp4 -vf fps=1/{clip_length/K} frame_%03d.jpg
```

Or via `decord` library (GPU-accelerated, much faster than OpenCV for batch processing):

```python
vr = VideoReader(path)
frames = vr.get_batch(sampled_indices).asnumpy()
```

**Step 2 — Face detection and alignment**

Use MTCNN (Zhang et al., 2016) or RetinaFace for robust multi-face detection:

- Detect all faces in each frame
- Select the face with highest confidence score (speaker is typically the speaking face, though this is imperfect in MELD)
- Align face to canonical 112×112 orientation using 5-point landmarks

**Libraries:** `facenet-pytorch` (MTCNN), `insightface` (RetinaFace + face alignment)

**Step 3A — Facial expression features**

Option A: **DINO-v2 ViT-B/14** fine-tuned face features

- Extract CLS token from DINOv2 ViT-B → 768-d
- DINOv2 has strong visual features that generalize well

Option B: **AffectNet-pretrained ResNet/EfficientNet**

- Use a ResNet50 or EfficientNet-B2 pre-trained on AffectNet (500k face images, 8 emotions)
- Extract penultimate layer: 1024-d or 2048-d
- **Library:** `deepface`, `fer`, or direct torchvision model

Option C: **Action Units (AU) via py-feat**

- Extract 20 action unit intensities per frame (AU1, AU2, AU4, ..., AU45)
- AUs are the foundational Ekman-based facial coding units
- **Library:** `py-feat` or `OpenFace2.0` (command-line, C++)
- Less VRAM-intensive, interpretable

**Step 3B — Scene / Global frame features**

Beyond faces, the scene conveys context (e.g., a hospital room suggests sadness, a party suggests joy):

Option: **CLIP ViT-L/14** (Radford et al., 2021)

- `clip-vit-large-patch14` from HuggingFace
- Extract 768-d visual embedding from full frame (not just face)
- CLIP's language-vision alignment makes it semantically rich

### Architecture: Dual-Stream Facial + Scene Fusion

**Input:**

- Face stream: $(K, d_f)$ tensor, one face embedding per frame
- Scene stream: $(K, d_s)$ tensor, one CLIP embedding per frame

**Step 1 — Per-stream temporal modeling**

For each stream, apply a Temporal Transformer or Bidirectional Mamba2:

$$H^F = \text{TemporalTransformer}(X^F), \quad H^S = \text{TemporalTransformer}(X^S)$$

Pool: $h^F = \text{CLS}(H^F)$, $h^S = \text{CLS}(H^S)$

**Step 2 — Cross-stream attention**

$$h^{V} = \text{CrossAttention}(Q = h^F, K = h^S, V = h^S) + h^F$$

This lets the facial stream query scene context. If no face is detected reliably, $h^F$ falls back to zero, letting $h^V = h^S$.

**Step 3 — Classification**

$$\hat{y} = \text{Softmax}(W_c h^V)$$

**Modular components:**

- Face detector: MTCNN → RetinaFace → YOLOv8-face
- Face features: AffectNet-ResNet → DINOv2 → CLIP → py-feat AUs
- Scene features: CLIP → ResNet50-ImageNet → SlowFast
- Temporal: Transformer → BiLSTM → Mamba2 → GRU
- Pool: CLS → Attentive → Mean

**Special handling: Missing faces**

In ~30% of MELD clips, no face or multiple faces make detection unreliable. Strategy:

- If no face detected: use zero vector for face stream, rely on scene stream only
- If multiple faces: aggregate as mean of all detected faces (or train a speaker-identification-guided selector)

---

## Task 4 — Pose-Only Emotion Recognition

### Context

Body pose is a weaker modality for MELD specifically (TV broadcast, often only head/shoulders visible) but contributes marginally in multimodal fusion. Target: ≥45% WF1.

### Feature Extraction

**MediaPipe Pose/Holistic:**

- `mediapipe.solutions.holistic` extracts 33 body landmarks + 468 face landmarks + 21 hand landmarks per frame
- For MELD (mostly talking-head shots), relevant landmarks: shoulders, neck, head tilt, face landmarks

Per frame, extract:

- 33 pose landmarks × 3 coordinates (x, y, z) = 99-d
- Optionally add face mesh landmarks for head orientation

Aggregate over $K$ frames: $(K, 99)$ tensor

**OpenPose (alternative):**

- Detects 18 body keypoints in COCO format
- More robust for partial body detection but heavier to run

### Architecture

**Input:** $(K, d_p)$ pose tensor, $d_p = 99$

**Step 1 — Temporal GRU:**

$$h_t^P, c_t = \text{GRU}(x_t^P, h_{t-1}^P)$$

Use 2-layer bidirectional GRU, hidden size 128. Last hidden state: $(2 \times 128,) = 256$-d.

**Why GRU over Transformer for pose?** Pose sequences are short ($K \leq 16$ frames) and have strong temporal locality. GRU is parameter-efficient and less prone to overfitting on small sequences.

**Step 2 — Graph Neural Network over skeleton (optional, stronger)**

Model the pose skeleton as a graph $G = (V, E)$ where $V$ = keypoints, $E$ = body-part connections (shoulder–elbow, elbow–wrist, etc.).

Apply **Spatial-Temporal Graph Convolutional Network (ST-GCN)**:

$$H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$$

where $\hat{A}$ is the normalized adjacency matrix of the skeleton graph, $H^{(l)}$ is the feature matrix at layer $l$.

This explicitly models how movements propagate through the body.

**Library:** `torch-geometric` or custom ST-GCN implementation

**Step 3 — Classification:** Same MLP head as other tasks.

---

## Task 5 — Text + Audio (Speech-Text) Emotion Recognition

### Mathematical Framework

Given text representation $h^T_i$ and audio representation $h^A_i$, the fusion produces:

$$h^{TA}_i = \text{Fuse}(h^T_i, h^A_i)$$

### Architecture A: Cross-Modal Attention Fusion

**Cross-modal attention** (Tsai et al., 2019, Multimodal Transformer):

Text-to-Audio attention: $$h^{T \to A} = \text{CrossAttn}(Q = H^T, K = H^A, V = H^A)$$

$$= \text{softmax}\left(\frac{H^T W_Q (H^A W_K)^T}{\sqrt{d_k}}\right) H^A W_V$$

Audio-to-Text attention: $$h^{A \to T} = \text{CrossAttn}(Q = H^A, K = H^T, V = H^T)$$

Final representation: $$h^{TA} = \text{Concat}(h^{T \to A}, h^{A \to T})$$

$$\hat{y} = \text{Softmax}(W_c h^{TA})$$

**Why cross-modal attention?** It allows each modality to query the other for complementary information. Audio can "ask" the text: "which words align with my emotional peaks?" Text can "ask" the audio: "how is this sentence being said?"

### Architecture B: Residual Gating (Late Fusion)

A simpler but robust approach:

$$h^{TA} = g \odot h^T + (1-g) \odot h^A, \quad g = \sigma(W_g [h^T; h^A])$$

This is a learned soft gate that weights modalities based on their reliability per sample. When audio is noisy, $g \to 1$, effectively relying on text.

**Why gating?** MELD audio is unreliable; a hard concatenation gives equal weight to noisy audio. Gating learns to downweight when one modality is uninformative.

### Architecture C: MulT (Multimodal Transformer, Tsai et al. 2019)

The full MulT architecture uses directional cross-modal attention in both directions, followed by a Transformer within each modality:

1. $H^{T \to A}$ = crossmodal transformer (T attends A)
2. $H^{A \to T}$ = crossmodal transformer (A attends T)
3. Concatenate + apply self-attention
4. Classify

This is the gold standard for 2-modality fusion. Pre-implemented at `https://github.com/yaohungt/Multimodal-Transformer`.

### Architecture D: Mamba2-based Sequence Fusion

Inspired by MaTAV, project both modalities to same hidden dimension $d$, then concatenate along sequence dimension and process with Mamba2:

$$X^{TA} = [H^T_{\text{seq}}; H^A_{\text{seq}}] \in \mathbb{R}^{(T+S) \times d}$$

$$H^{TA} = \text{Mamba2}(X^{TA})$$

Pool the final hidden state or apply mean pooling. Mamba2 handles long-range dependencies with linear complexity, useful when both modalities have long sequences.

---

## Task 6 — Text + Video Emotion Recognition

### Architecture: CLIP-Text Alignment + Cross-Attention

**Motivation:** Text and video can be aligned in CLIP's joint embedding space, providing a semantically consistent fusion ground.

**Step 1 — Aligned projection:**

$$\tilde{h}^T = W_{\text{proj}}^T h^T, \quad \tilde{h}^V = W_{\text{proj}}^V h^V, \quad \tilde{h}^T, \tilde{h}^V \in \mathbb{R}^d$$

**Step 2 — Modality alignment loss (pre-training on MELD):**

$$\mathcal{L}_{\text{align}} = -\text{CosSim}(\tilde{h}^T, \tilde{h}^V) \cdot \mathbb{1}[\text{same emotion}]$$

This encourages representations of the same utterance in text and video to be close in the shared space.

**Step 3 — Fusion via cross-attention + residual:**

$$h^{TV} = h^T + \text{CrossAttn}(Q = h^T, K = h^V, V = h^V)$$

**Note on MELD Text+Video:** This combination may not significantly outperform text-only because video is noisy. The alignment loss helps regularize the video representation toward the more reliable text space.

---

## Task 7 — Audio + Video Emotion Recognition

### Architecture: Temporal Cross-Modal Fusion

This combination is interesting because both modalities are temporal and can be aligned frame-by-frame.

**Frame-level alignment:**

Given audio frames ${a_1, \ldots, a_T}$ and video frames ${v_1, \ldots, v_K}$ (possibly different $T, K$), upsample/downsample to common resolution $M$ via linear interpolation.

**Fused temporal sequence:**

$$x_t^{AV} = [a_t; v_t] \in \mathbb{R}^{d_A + d_V}$$

Apply Temporal Transformer:

$$H^{AV} = \text{Transformer}(X^{AV})$$

Pool → classify. Alternatively, apply MaTAV's Mamba architecture here.

**Contrastive alignment loss:**

$$\mathcal{L}_{\text{MEC}} = -\sum_i \log \frac{\exp(s(\hat{a}_i, \hat{v}_i) / \tau)}{\sum_j \exp(s(\hat{a}_i, \hat{v}_j) / \tau)}$$

This Multimodal Emotion Contrastive (MEC) Loss from MaTAV aligns same-utterance audio and video representations, pulling apart different utterances even if they share similar acoustic/visual properties from different emotion classes.

---

## Task 8 — Text + Audio + Video Emotion Recognition

### Architecture A: MaTAV-Inspired Three-Stream Fusion

This is the canonical trimodal architecture. We propose enhancements over the base MaTAV.

**Step 1 — Modality-specific encoders:**

|Modality|Encoder|Output dim|
|---|---|---|
|Text|RoBERTa-Large or LLM projected|$d_T = 1024$|
|Audio|WavLM-Large + Conformer|$d_A = 256$|
|Video|DINOv2-Face + CLIP-Scene|$d_V = 512$|

Project all to common $d = 256$ via linear layers.

**Step 2 — Cross-modal alignment (MEC-Loss):**

$$\mathcal{L}_{\text{MEC}} = \mathcal{L}_{\text{align}}(T, A) + \mathcal{L}_{\text{align}}(T, V) + \mathcal{L}_{\text{align}}(A, V)$$

Each pairwise alignment loss uses contrastive learning across the batch.

**Step 3 — Mamba2-based multimodal fusion:**

Concatenate projected features along sequence:

$$X^{TAV} = [h^T_1, \ldots, h^T_{L_T}; h^A_1, \ldots, h^A_{L_A}; h^V_1, \ldots, h^V_{L_V}]$$

Apply Bidirectional Mamba2:

$$\vec{H} = \overrightarrow{\text{Mamba2}}(X^{TAV}), \quad \overleftarrow{H} = \overleftarrow{\text{Mamba2}}(X^{TAV})$$

$$H = \vec{H} + \overleftarrow{H}$$

Pool final representation: $h^{TAV} = \text{AttentivePool}(H)$

**Mamba2 state-space equations:**

The discretized SSM (Mamba) recurrence:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$

where: $$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

$\Delta$ is a data-dependent selection parameter — the key innovation of Mamba over S4. This allows the model to "select" which information to carry in the hidden state based on input content, mimicking attention-like behavior with $O(N)$ complexity (vs $O(N^2)$ for full attention).

**Library:** `mamba-ssm` pip package (from Dao et al. 2024, requires CUDA toolkit, available on Kaggle)

**Step 4 — Classification with auxiliary tasks:**

Main task: 7-class emotion classification Auxiliary task: 3-class sentiment classification (shares backbone, separate head)

$$\mathcal{L} = \mathcal{L}_{\text{focal}}^{\text{emotion}} + \lambda_s \mathcal{L}_{\text{CE}}^{\text{sentiment}} + \lambda_{\text{MEC}} \mathcal{L}_{\text{MEC}}$$

Auxiliary sentiment task regularizes the shared representation and is easy to solve, helping with gradient stability on the hard emotion task.

### Architecture B: Hierarchical Cross-Attention (MultiMamba variant, enhanced)

This is your thesis architecture improved:

**Level 1 — Intra-modal temporal modeling:** Each modality goes through its own Mamba2 encoder to learn temporal patterns independently.

**Level 2 — First cross-attention (TAV):**

Given $h^T, h^A, h^V$ as query, key, value respectively (rotating roles):

$$\text{CA}_{T \to AV} = \text{CrossAttn}(Q = h^T, K = [h^A; h^V], V = [h^A; h^V])$$

**Level 3 — Joint representation:**

$$h^{\text{joint}} = \text{Concat}(h^T, h^A, h^V) + \text{Dropout}$$

**Level 4 — Second cross-attention with joint:**

$$h^{\text{fused}} = \text{CrossAttn}(Q = \text{CA}_{T \to AV}, K = h^{\text{joint}}, V = h^{\text{joint}})$$

**Level 5 — Mamba2 contextual integration:**

$$h^{\text{ctx}} = \text{Mamba2}(h^{\text{fused}}_{\text{seq}})$$

This is your MultiMamba architecture but with: (a) independent Mamba2 pre-encoders per modality instead of CNN, (b) richer cross-attention levels, (c) MEC-Loss alignment.

---

## Task 9 — Full Multimodal Fusion

This task brings together all modalities: text, audio, video, pose, speaker identity, and explicit dialogue context. This is the main system.

### 9.1 Additional Input: Speaker Identity

MELD has named speakers (Ross, Monica, Chandler, Joey, Rachel, Phoebe + recurring/guest characters). Speaker identity is a strong prior:

**Option A: Learnable speaker embeddings**

$$e^{sp} = E[s_i] \in \mathbb{R}^{d_{sp}}, \quad E \in \mathbb{R}^{|\mathcal{S}| \times d_{sp}}$$

Where $|\mathcal{S}|$ = number of unique speakers. For MELD, 6 main characters + ~100 others → use embedding for known speakers, UNK embedding for others.

**Option B: Speaker description (from LLM, pre-extracted)**

As in BiosERC: 100-word description of speaker personality → encode with sentence-BERT → 768-d vector.

This is superior for generalization: if an unknown speaker appears in test set, their description from dialogue context is still meaningful.

**Option C: Speaker voice embedding (i-vector / x-vector)**

Pre-trained speaker recognition model (e.g., pyannote.audio's `wespeaker-voxceleb-resnet34-LM`) extracts a speaker embedding from audio, independent of emotion. This captures speaker-specific acoustic style.

### 9.2 Full Architecture: ConvERSATION-Aware Multimodal Encoder

**All input streams per utterance $i$:**

|Stream|Shape|Description|
|---|---|---|
|$H^T_i$|$(L_T, 1024)$|Text tokens from RoBERTa/LLM|
|$h^A_i$|$(256,)$|Audio utterance embedding|
|$h^V_i$|$(512,)$|Video utterance embedding|
|$h^P_i$|$(128,)$|Pose utterance embedding|
|$e^{sp}_i$|$(128,)$|Speaker embedding|
|$h^{bio}_i$|$(768,)$|Speaker biography SBERT embedding|

**Step 1 — Modality projection to $d = 256$:**

$$\tilde{h}^m_i = \text{LayerNorm}(\text{Linear}(h^m_i)) \quad \forall m \in {T, A, V, P, sp, bio}$$

**Step 2 — Dialogue context modeling:**

Stack representations across context window $w$:

$$X^m = [\tilde{h}^m_{i-w}; \ldots; \tilde{h}^m_i] \in \mathbb{R}^{(w+1) \times d}$$

Apply a Temporal Transformer (or Mamba2) per modality to get context-aware representations:

$$\tilde{H}^m = \text{TemporalEncoder}(X^m), \quad \tilde{h}^m_{\text{ctx}} = \tilde{H}^m[-1]$$

(Last position = current utterance, attending to history)

**Step 3 — Cross-modal fusion block:**

**First cross-attention** — Text as query, all other modalities as key/value:

$$h^{T \leftarrow {A,V,P}} = \text{MHA}(Q = \tilde{h}^T_{\text{ctx}}, K = [\tilde{h}^A; \tilde{h}^V; \tilde{h}^P], V = [\tilde{h}^A; \tilde{h}^V; \tilde{h}^P])$$

**Second cross-attention** — All modalities attend to speaker:

$$h^{m \leftarrow sp} = \tilde{h}^m + \text{Attn}(Q = \tilde{h}^m, K = e^{sp} \oplus h^{bio}, V = e^{sp} \oplus h^{bio})$$

**Step 4 — Modality gating with uncertainty estimation:**

Inspired by the Dropkey approach: use a learned confidence gate per modality:

$$g^m = \sigma(W_g^m h^m_{\text{ctx}}), \quad h^m_{\text{gated}} = g^m \odot h^m_{\text{ctx}}$$

Sum with residual: $h^{\text{fused}} = \sum_m h^m_{\text{gated}} + h^T_{\text{ctx}}$ (text always residual)

**Step 5 — Mamba2 sequence over fused representations:**

Process the fused sequence across all $w+1$ dialogue positions:

$$H^{\text{dial}} = \text{Mamba2}([h^{\text{fused}}_{i-w}; \ldots; h^{\text{fused}}_i])$$

$$h^{\text{final}} = H^{\text{dial}}[-1]$$

**Step 6 — Classification:**

$$\hat{y}^{\text{emotion}} = \text{Softmax}(W^e h^{\text{final}})$$ $$\hat{y}^{\text{sentiment}} = \text{Softmax}(W^s h^{\text{final}})$$

**Loss:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}^{e} + 0.3 \cdot \mathcal{L}_{\text{CE}}^{s} + 0.1 \cdot \mathcal{L}_{\text{MEC}} + 0.05 \cdot \mathcal{L}_{\text{SupCon}}$$

All loss weights are modular hyperparameters to tune.

### 9.3 Modality Dropout for Robustness

During training, randomly drop entire modalities with probability $p_{\text{drop}} = 0.15$ per modality (independently). This:

1. Forces the model to work without any single modality
2. Simulates test-time failure modes (no face detected, corrupted audio)
3. Acts as a regularizer
4. Allows the model to be deployed with partial modalities

### 9.4 Task 9 Training Schedule

```
Phase 1: Pretrain each modality encoder independently (Tasks 1-4)
Phase 2: Freeze encoders, train fusion module only (5 epochs)
Phase 3: Unfreeze all, fine-tune end-to-end with small LR (5 epochs)
Phase 4: Apply curriculum learning based on dialogue difficulty
```

---

## 14. Class Imbalance: Strategies Per Task

This is the most critical practical issue in MELD. Neutral dominates at 47%, fear/disgust appear at 2.7% each.

### 14.1 Loss Function Comparison

|Strategy|Formula|Best For|
|---|---|---|
|Standard CE|$-\sum y_c \log \hat{p}_c$|Balanced datasets|
|Weighted CE|$-\sum w_c y_c \log \hat{p}_c$|Moderate imbalance|
|Focal Loss|$-(1-\hat{p})^\gamma \log \hat{p}$|Hard examples|
|Weighted Focal|$-w_c (1-\hat{p})^\gamma \log \hat{p}$|**MELD — recommended**|
|Label Smoothing|Replace $y$ with $(1-\epsilon)y + \epsilon/C$|Overconfidence|
|SupCon|Contrastive on embeddings|Rare class representation|

**Recommended combination:** Weighted Focal + Label Smoothing ($\epsilon=0.1$) + SupCon auxiliary:

$$\mathcal{L} = \mathcal{L}_{\text{wFocal}} + 0.05 \cdot \mathcal{L}_{\text{smooth}} + 0.2 \cdot \mathcal{L}_{\text{SupCon}}$$

### 14.2 Class Weight Calculation

$$w_c = \left(\frac{N_{\max}}{N_c}\right)^{0.5}$$

Square-root damping avoids over-correcting (full inverse frequency over-emphasizes fear/disgust to the point of degrading neutral precision).

|Emotion|$N_c$|$w_c$ (sqrt-inverse)|
|---|---|---|
|neutral|4710|1.0|
|joy|1743|1.64|
|anger|1109|2.06|
|surprise|1205|1.98|
|sadness|683|2.62|
|disgust|268|4.19|
|fear|268|4.19|

### 14.3 Data Augmentation for Text

For rare classes (fear, disgust), augment training samples:

**Strategy A: LLM paraphrase augmentation**

Use Qwen3-14B (via API or local) to generate paraphrases of fear/disgust utterances while preserving emotion:

```
Prompt: "Rewrite the following utterance in 3 different ways, preserving the {emotion} emotion:
Utterance: '{original}'
Maintain the same emotional content, vary phrasing only."
```

Generate 5× more fear/disgust samples. Verify quality with the same LLM.

**Strategy B: Synonym/back-translation augmentation**

Use `nlpaug` library:

- `nlpaug.augmenter.word.ContextualWordEmbsAug` (BERT-based contextual replacement)
- Back-translation: English → French → English via Helsinki-NLP models

**Strategy C: Mixup in embedding space**

$$\tilde{h} = \lambda h_{x_i} + (1-\lambda) h_{x_j}, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

Apply in the penultimate embedding space, not raw text. Particularly useful for intermediate classes.

### 14.4 Data Augmentation for Audio

- **SpecAugment**: Mask frequency bands and time steps of the mel-spectrogram (Park et al., 2019)
- **Noise injection**: Add Gaussian noise at SNR 20–40 dB, simulating recording variation
- **Pitch shift**: ±2 semitones (preserves speech intelligibility, changes prosody)
- **Time stretch**: ±10% speed (via `librosa.effects.time_stretch`)

---

## 15. Modular Hyperparameter Protocol

### 15.1 Hyperparameter Space per Task

All tasks share a common modular framework. Swapping a component triggers re-tuning only of that module's hyperparameters.

**Universal hyperparameters (all tasks):**

|Hyperparameter|Search Space|Method|
|---|---|---|
|Learning rate|[1e-5, 5e-4]|Log-uniform, Optuna|
|Batch size|{8, 16, 32, 64}|Grid|
|Dropout|[0.05, 0.4]|Uniform|
|Context window $w$|{3, 5, 7, 10}|Grid|
|Focal $\gamma$|{1.0, 1.5, 2.0, 2.5}|Grid|
|Class weight power|{0.5, 0.75, 1.0}|Grid|
|Label smoothing $\epsilon$|{0.0, 0.05, 0.1, 0.15}|Grid|
|LoRA rank $r$|{8, 16, 32}|Grid (Task 1 only)|

### 15.2 Optuna Integration

Use Optuna with TPE (Tree-structured Parzen Estimator) sampler and Hyperband pruner:

```python
# Pseudocode — not implementation
study = optuna.create_study(
    direction="maximize",  # WF1
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.HyperbandPruner()
)
study.optimize(objective_fn, n_trials=50)
```

For Kaggle (limited time), use 20–30 trials per task with early stopping (patience=5).

### 15.3 Cross-Validation Strategy

MELD provides a fixed dev/test split. Do **not** use test set for hyperparameter selection.

Procedure:

1. All hyperparameter tuning → `dev_sent_emo.csv`
2. Best config → single run on `test_sent_emo.csv`
3. Report mean ± std over 5 random seeds (different LoRA init) for final results

### 15.4 Component Swap Registry

Track all experiments in a structured YAML or W&B config:

```yaml
task: 1
backbone: "roberta-large"
context_window: 5
loss: "weighted_focal"
focal_gamma: 2.0
class_weight_power: 0.5
lr: 3e-4
batch_size: 16
curriculum: true
augmentation: ["paraphrase", "mixup"]
retrieval_repo: true
speaker_bio: true
explicit_emotion: true
implicit_emotion: true
dev_wf1: 71.2
test_wf1: 70.8
```

Use `wandb` (free tier) or MLflow for tracking.

---

## 16. Feature Extraction Pipeline & Storage Strategy

### 16.1 Processing Order & Dependencies

```
raw .mp4 + .wav + .csv
    │
    ├── Text extraction:
    │     CSV Utterance column → already have text
    │     Context: group by Dialogue_ID, sort by Utterance_ID
    │     Speaker bio: Qwen3-14B API call per dialogue
    │     Explicit/implicit: Qwen3-14B API call per utterance
    │     Retrieval embeddings: SBERT encode all training utterances
    │
    ├── Audio extraction:
    │     .wav → resample 16kHz → WavLM-Large → pool → (1024,) per utterance
    │     Save: utterance_audio_features.npy  [N × 1024]
    │
    ├── Video extraction:
    │     .mp4 → sample K=8 frames → 
    │     Face: MTCNN detect → AffectNet-ResNet → (K, 1024) per utterance
    │     Scene: CLIP → (K, 768) per utterance
    │     Pool over K: (1024,), (768,)
    │     Save: utterance_face_features.npy, utterance_scene_features.npy
    │
    └── Pose extraction:
          .mp4 → sample K=8 frames → MediaPipe Holistic → (K, 99) per utterance
          Pool over K: (99,)
          Save: utterance_pose_features.npy
```

### 16.2 Storage Format

```
features/
├── train/
│   ├── text_roberta.npy          [9989 × 1024]
│   ├── audio_wavlm.npy           [9989 × 1024]
│   ├── video_face.npy            [9989 × 1024]
│   ├── video_scene_clip.npy      [9989 × 768]
│   ├── pose_mediapipe.npy        [9989 × 99]
│   ├── speaker_bio_sbert.npy     [9989 × 768]
│   ├── speaker_id.npy            [9989]  (int)
│   ├── explicit_emotion.txt      [9989 lines]
│   ├── implicit_emotion.txt      [9989 lines]
│   └── retrieval_embeddings.npy  [9989 × 768]
├── dev/   (same structure, 1109 samples)
└── test/  (same structure, 2610 samples)
```

**Indexing**: Row $i$ in each `.npy` corresponds to `Sr No.` $i$ in the CSV. Maintain a mapping CSV: `features/index_map.csv` with `Sr No., split, dia_id, utt_id, emotion, sentiment, speaker`.

### 16.3 Batch Loading

During training, a `DataLoader` loads one batch = $B$ utterances with their features:

```
batch = {
    "text_emb":    Tensor[B, 1024],
    "audio_emb":   Tensor[B, 1024],
    "video_face":  Tensor[B, 1024],
    "video_scene": Tensor[B, 768],
    "pose":        Tensor[B, 99],
    "speaker_bio": Tensor[B, 768],
    "speaker_id":  Tensor[B],
    "context_mask": Tensor[B, w+1],  # which context positions are valid
    "label_emotion": Tensor[B],
    "label_sentiment": Tensor[B],
}
```

Context window: use a collation function that groups utterances from the same dialogue, pads shorter dialogues.

### 16.4 Local vs Cloud Processing

|Feature|Local (1650 Super, 4GB)|Kaggle/Colab|
|---|---|---|
|Text (RoBERTa encode)|✓ (base model only)|✓|
|Audio (WavLM-Large)|✓ (batch size=1)|✓ (batch=8)|
|Audio (emotion2vec)|✓|✓|
|Video (MediaPipe)|✓ (CPU, slow)|✓|
|Video (CLIP ViT-L)|✗ (5GB+ VRAM)|✓|
|Video (AffectNet-ResNet)|✓|✓|
|LLM feature extraction (Qwen3-14B)|✗ (too large)|✓ (Groq API: free)|
|LLM fine-tuning (7B + LoRA)|✗|✓|

**Recommendation:** Use Groq API (`api.groq.com`) for Qwen3/Llama3 inference — free tier, very fast, sufficient for one-time feature extraction.

---

## 17. Evaluation Protocol

### 17.1 Primary Metric

**Weighted F1** (WF1) as used in all MELD papers:

$$\text{WF1} = \frac{\sum_c N_c \cdot F1_c}{\sum_c N_c}$$

computed on test set via `sklearn.metrics.f1_score(y_true, y_pred, average='weighted')`.

### 17.2 Per-Class Reporting

Always report per-class F1 alongside WF1. This shows whether fear/disgust are being learned. A model with WF1=70% that has F1=0% for fear is less useful than one with WF1=69% but F1=30% for fear.

### 17.3 Significance Testing

For comparing two configurations A and B: use **paired bootstrap test** (Efron & Tibshirani, 1993):

- Sample 1000 bootstrap subsets of the test set
- Compute WF1 for each subset under both systems
- $p$-value = fraction of bootstrap samples where B beats A

This is more appropriate than t-test for classification tasks.

### 17.4 Ablation Study Template

For each task, run ablations:

|Config|WF1|ΔWF1 vs Full|
|---|---|---|
|Full model|71.0|—|
|w/o speaker bio|70.1|-0.9|
|w/o explicit emotion|69.5|-1.5|
|w/o implicit emotion|70.3|-0.7|
|w/o retrieval|70.5|-0.5|
|w/o curriculum learning|70.6|-0.4|
|w/o class weighting|68.0|-3.0|

---

## 18. Recommended Libraries & Dependencies

### 18.1 Core Framework

```
torch>=2.0.0              # PyTorch - all deep learning
transformers>=4.40.0       # HuggingFace: RoBERTa, LLaMA, WavLM, Qwen
peft>=0.10.0               # LoRA, QLoRA, PEFT
accelerate>=0.28.0         # Multi-GPU, mixed precision
datasets>=2.18.0           # HuggingFace datasets format
```

### 18.2 Feature Extraction

```
soundfile>=0.12            # WAV loading
librosa>=0.10              # Audio feature extraction, resampling
opensmile>=2.5             # Handcrafted audio features (LLD/HSF)
decord>=0.6.0              # Fast video frame reading (GPU-accelerated)
opencv-python>=4.9         # Image processing
facenet-pytorch>=2.6       # MTCNN face detection
insightface>=0.7           # RetinaFace, ArcFace
mediapipe>=0.10            # Pose + holistic estimation
clip @ git+https://github.com/openai/CLIP.git  # CLIP features
```

### 18.3 LLM & NLP

```
sentence-transformers>=2.7  # SBERT retrieval embeddings
faiss-cpu or faiss-gpu      # Approximate nearest neighbor retrieval
groq                        # Groq API for Qwen/Llama inference
tokenizers>=0.19            # Fast tokenization
```

### 18.4 Training & Experiment Management

```
optuna>=3.6               # Hyperparameter optimization
wandb>=0.17               # Experiment tracking (free tier)
tqdm>=4.66                # Progress bars
numpy>=1.26               # Numerical arrays
pandas>=2.2               # CSV loading
scikit-learn>=1.4         # Metrics (WF1), class weights
```

### 18.5 Mamba Architecture

```
mamba-ssm>=2.2.2          # Mamba2 state-space model (requires CUDA)
causal-conv1d>=1.4.0      # Required by mamba-ssm
```

**Note on Mamba on Kaggle:** `mamba-ssm` requires CUDA toolkit >= 11.6. Kaggle P100/T4 environments support this. Install with `pip install mamba-ssm causal-conv1d --no-build-isolation`.

If Mamba installation fails: substitute with `flash-attn` based Transformer, which has similar efficiency characteristics on MELD's sequence lengths.

### 18.6 QLoRA for 4-bit Quantization (memory savings)

```
bitsandbytes>=0.43         # 4-bit/8-bit quantization (QLoRA)
```

QLoRA (Dettmers et al., 2023) allows fine-tuning a 7B model in 4-bit NF4 quantization, reducing VRAM to ~5GB — viable on Kaggle P100 with full sequence length.

---

## 19. References

**Datasets & Benchmarks:**

- Poria et al. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. ACL 2019. arXiv:1810.02508
- Busso et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. LREC.

**ERC Systems (Text):**

- Lei et al. (2023). InstructERC: Reforming ERC with a Retrieval Multi-Task LLMs Framework. arXiv:2309.11911
- Xue et al. (2024). BiosERC: Integrating Biography Speakers Supported by LLMs for ERC Tasks. arXiv:2407.04279
- Fu et al. (2025). LaERC-S: Improving LLM-based ERC with Speaker Characteristics. arXiv:2403.07260
- Li et al. (2025). Do LLMs Feel? PRC-Emo: Prompts, Retrieval, Curriculum Learning. AAAI 2026. arXiv:2511.07061

**ERC Systems (Multimodal):**

- Li et al. (2024). MaTAV: Mamba-Enhanced Text-Audio-Video Alignment for ERC. arXiv:2409.05243
- Zhang et al. (2024). DER-GCN: Dialog and Event Relation-Aware GCN for Multimodal ERC. IEEE TNNLS.
- Nguyen et al. (2024). MultiDAG+CL: Curriculum Learning Meets DAG for Multimodal ERC. LREC-COLING 2024.
- Zhou et al. (2025). SALM: Sparse Alignment and Liquid-Mamba for Multimodal ERC. Electronics 2025.

**Core Architectures:**

- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Gu & Dao (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. COLM 2024.
- Dao & Gu (2024). Transformers are SSMs: Mamba-2. ICML 2024.
- Touvron et al. (2023). LLaMA 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288
- Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.

**Audio:**

- Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. IEEE JSTSP.
- Gulati et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. INTERSPEECH 2020.
- Eyben et al. (2010). OpenSMILE: The Munich Versatile and Fast Open-Source Audio Feature Extractor. ACM MM 2010.

**Vision:**

- Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
- Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193
- Zhang et al. (2016). Joint Face Detection and Alignment Using MHCNN. IEEE SPL. [MTCNN]
- Mollahosseini et al. (2019). AffectNet: A Database for Facial Expression Recognition. IEEE TAC.
- Lugaresi et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172

**Loss & Training:**

- Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.
- Khosla et al. (2020). Supervised Contrastive Learning. NeurIPS 2020.
- Bengio et al. (2009). Curriculum Learning. ICML 2009.
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.

**Speaker Modeling:**

- Ghosal et al. (2020). COSMIC: Commonsense Knowledge for Emotion Identification in Conversations. EMNLP 2020.
- Bosselut et al. (2019). COMET: Commonsense Transformers for Knowledge Graph Construction. ACL 2019.
- Sap et al. (2019). ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning. AAAI 2019.

---

_Document version: April 2026. All SOTA numbers as of April 2026 based on published arxiv/conference results._
# Speaker Diarization Clustering & Scoring Algorithms Research

Research conducted 2026-03-21. Focus: preventing over-segmentation in online speaker diarization.

**Problem statement**: VoxTerm's current approach (online cosine similarity with threshold 0.30, EMA centroid update alpha=0.95) turns 2 real speakers into 5+ detected speakers. Root causes: embedding noise, static threshold, no cluster merging, no probabilistic scoring.

---

## Table of Contents

1. [PLDA Scoring](#1-plda-probabilistic-linear-discriminant-analysis-scoring)
2. [VBx (Bayesian HMM)](#2-vbx-variational-bayes-hmm-clustering)
3. [Spectral Clustering](#3-spectral-clustering)
4. [Agglomerative Hierarchical Clustering (AHC)](#4-agglomerative-hierarchical-clustering-ahc)
5. [EEND (End-to-End Neural Diarization)](#5-eend-end-to-end-neural-diarization)
6. [Sticky HDP-HMM](#6-sticky-hdp-hmm)
7. [UIS-RNN](#7-uis-rnn-unbounded-interleaved-state-rnn)
8. [NME-SC (Auto-Tuning Spectral Clustering)](#8-nme-sc-auto-tuning-spectral-clustering)
9. [BIC-Based Clustering](#9-bic-based-clustering)
10. [Cluster Purification / Cross-EM Refinement](#10-cluster-purification--cross-em-refinement)
11. [Online Incremental Clustering (Current SOTA)](#11-online-incremental-clustering-current-approaches)
12. [Pyannote Pipeline](#12-pyannote-speaker-diarization-pipeline)
13. [Cross-Cutting Concerns](#13-cross-cutting-concerns)
14. [Recommendations for VoxTerm](#14-recommendations-for-voxterm)

---

## 1. PLDA (Probabilistic Linear Discriminant Analysis) Scoring

**Key references**:
- "Speaker diarization with PLDA i-vector scoring and unsupervised calibration" (IEEE 2014)
- "Probabilistic Back-ends for Online Speaker Recognition and Clustering" (ICASSP 2023, arXiv:2302.09523)
- "Scoring of Large-Margin Embeddings for Speaker Verification: Cosine or PLDA?" (Interspeech 2022, arXiv:2204.03965)
- "Unifying Cosine and PLDA Back-ends for Speaker Verification" (arXiv:2204.10523)

**How it works**:
PLDA models speaker embeddings as a combination of a speaker-specific component and a session/channel-specific noise component. It computes a log-likelihood ratio between two hypotheses: "these embeddings come from the same speaker" vs. "these come from different speakers." Mathematically, PLDA separately models across-class (between-speaker) and within-class (within-speaker) variability, emphasizing speaker-discriminative dimensions while de-emphasizing confusable information.

Key insight: cosine similarity is a special case of PLDA scoring. By constraining the PLDA covariance matrices in specific ways, you recover cosine scoring. PLDA adds the ability to model and discount noise.

**Online or offline**: Both. The ICASSP 2023 paper specifically presents an "extremely constrained version of PLDA" designed for online speaker clustering. It handles multi-enrollment scoring (comparing a new embedding against multiple enrolled embeddings of a speaker) more gracefully than cosine scoring, because cosine scoring is poorly calibrated when the number of enrollment utterances varies.

**How it handles over-segmentation**:
- PLDA produces calibrated log-likelihood ratios, not raw distances. This means the same threshold has consistent meaning regardless of embedding quality or number of observations.
- Multi-enrollment PLDA naturally handles uncertainty: more observations of a speaker reduce uncertainty, making the decision to merge more confident.
- Within-class covariance modeling discounts embedding noise that causes same-speaker embeddings to look different.

**Computational cost**:
- Scoring is O(D^2) where D is embedding dimension (typically 192 for ECAPA-TDNN). For 192-dim embeddings, this is a small matrix operation (~37K multiply-adds). Easily under 1ms per comparison.
- Training the PLDA model requires a dataset of speaker-labeled embeddings (VoxCeleb works).

**Needs training data**: Yes, to estimate the between-speaker and within-speaker covariance matrices. However, a pre-trained PLDA model from Kaldi/SpeechBrain can be used directly. The "extremely constrained PLDA" from the ICASSP 2023 paper has very few parameters and is easier to train.

**Open-source implementations**:
- SpeechBrain: `speechbrain.processing.PLDA_LDA` module
- Kaldi CALLHOME recipe: includes pre-trained PLDA models
- GitHub: `prachiisc/PLDA_scoring` (Python implementation with pretrained model)

**Key insight vs. naive cosine + threshold**:
PLDA models the noise in embeddings explicitly. When the same speaker produces two slightly different embeddings, PLDA recognizes this as expected within-speaker variability rather than evidence of different speakers. Cosine similarity treats all dimensions equally; PLDA learns which dimensions are speaker-discriminative vs. noise.

**Practical finding**: With modern large-margin neural embeddings (like ECAPA-TDNN), cosine scoring and PLDA perform comparably in matched-domain conditions. However, PLDA provides 10.9% EER reduction and 4.9% minDCF reduction when the within-speaker covariance is properly constrained. PLDA excels in domain mismatch scenarios.

---

## 2. VBx (Variational Bayes HMM Clustering)

**Key references**:
- Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization" (Computer Speech & Language 2022, arXiv:2012.14952)
- "Discriminative Training of VBx Diarization" (arXiv:2310.02732)
- GitHub: `BUTSpeechFIT/VBx`

**How it works**:
VBx is a three-stage approach: (1) extract x-vectors/embeddings from short segments, (2) initialize clusters via AHC, (3) refine using a Bayesian HMM. The HMM models speaker turns as hidden states and embeddings as emissions. Each speaker's emission distribution is a Gaussian parameterized by a PLDA model. Variational Bayes inference iteratively:
- E-step (forward-backward): compute posterior probability of each segment belonging to each speaker, using transition probabilities that encode speaker continuity.
- M-step: update speaker-specific parameters (means in PLDA space) based on soft assignments.

**Key parameters**:
- **Fa** (scaling factor for acoustic scores): controls how strongly embeddings influence cluster assignment. Typical value: 1.
- **Fb** (scaling factor for the prior/regularization): controls the penalty for creating new speakers. Higher Fb = fewer speakers. Typical values: 17-20 for CALLHOME. Most impactful parameter.
- **loopP** (self-transition probability): probability of staying in the same speaker state. Higher loopP = stronger continuity prior = less over-segmentation. Typical value: 0.99 (meaning only 1% chance of switching speakers at each frame boundary).
- **init_smoothing**: smoothing applied to initial AHC clusters. Typical value: 5.0.
- **AHC threshold**: tuned to under-cluster (set loose) so VBx has room to find optimal speaker count.

**Online or offline**: Offline. The forward-backward algorithm requires the full sequence. However, it can be applied to buffered windows of audio.

**How it handles over-segmentation**:
- **loopP = 0.99**: The transition matrix strongly biases toward staying with the current speaker. This directly prevents rapid switching / fragmentation.
- **Fb parameter**: High Fb values penalize creating additional speaker models, effectively regularizing toward fewer speakers.
- **Bayesian inference**: Speakers with too few observations will be absorbed into better-supported speakers during VB iterations.
- **PLDA emission model**: Same benefits as PLDA scoring -- models within-speaker variability.

**Computational cost**:
- Forward-backward scales as O(T * K^2) where T = number of frames and K = number of initial speaker candidates. For a typical meeting (T=1000 segments, K=10 init speakers), this is ~100K operations per iteration, with 10-20 iterations. Total: ~1-2M operations. Well under 200ms.
- Caveat: AHC initialization on recordings > 30 minutes can become very slow. A random initialization option is available that is substantially faster.

**Needs training data**: Needs a pre-trained PLDA model (between-speaker and within-speaker covariance matrices), typically from VoxCeleb. The VBx repo includes pre-trained models.

**Open-source implementations**:
- `BUTSpeechFIT/VBx` (Python, official)
- `BUTSpeechFIT/DVBx` (discriminatively trained variant)

**Key insight vs. naive cosine + threshold**:
VBx jointly optimizes speaker assignment across the entire recording. Instead of making independent decisions per segment ("is this cosine > 0.30?"), it considers: what is the globally most likely speaker assignment given all the embeddings and the constraint that speakers tend to persist? The transition probability (loopP) directly encodes the prior that speakers don't change every second -- exactly what prevents over-segmentation.

---

## 3. Spectral Clustering

**Key references**:
- Wang et al., "Speaker Diarization with LSTM" (Google, 2018, arXiv:1710.10468)
- Park & Han, "Auto-Tuning Spectral Clustering for Speaker Diarization Using NME" (IEEE SPL 2020, arXiv:2003.02405)
- "Self-Tuning Spectral Clustering for Speaker Diarization" (arXiv:2410.00023)
- GitHub: `wq2012/SpectralCluster` (Google)

**How it works**:
1. Compute pairwise cosine similarity matrix (affinity matrix) from all segment embeddings.
2. Apply refinement operations to the affinity matrix:
   - **Gaussian blur** (sigma=1): smooths noisy similarities.
   - **Row-wise thresholding** (p_percentile=0.95): sparsifies the matrix, keeping only strong connections.
   - **CropDiagonal**: replaces diagonal with max non-diagonal value per row.
   - **Symmetrization**: ensures the affinity matrix is symmetric.
3. Compute graph Laplacian from the refined affinity matrix.
4. Eigendecompose the Laplacian.
5. Estimate number of speakers using **eigengap analysis** (largest gap between consecutive eigenvalues).
6. Apply k-means to the top eigenvectors.

**Online or offline**: Primarily offline (needs all embeddings at once). However, Google's `SpectralCluster` library includes `MultiStageClusterer` for **streaming**: feed one embedding at a time to `streaming_predict`, which returns (corrected) labels for all previous embeddings. It uses compression thresholds (L=50, U1=200, U2=400) to keep memory bounded.

**How it handles over-segmentation**:
- Affinity matrix refinement (especially Gaussian blur and thresholding) smooths out noisy similarities that would create spurious clusters.
- Eigengap-based speaker count estimation inherently favors fewer clusters when the evidence for more is weak.
- Row-wise thresholding keeps only the strongest connections, preventing noise from creating false cluster boundaries.

**Computational cost**:
- Affinity matrix: O(N^2 * D) for N segments, D-dim embeddings.
- Eigendecomposition: O(N^3) in general, but sparse methods can reduce this.
- For 100 segments: eigendecomposition is trivial (<50ms). For 1000+ segments: can take seconds.
- Streaming version keeps N small via compression.

**Needs training data**: No. Operates directly on cosine similarities. No PLDA or other trained model needed. This is a major practical advantage.

**Open-source implementations**:
- `wq2012/SpectralCluster` (Python, Google, actively maintained)
- `tango4j/Auto-Tuning-Spectral-Clustering` (NME-SC, now maintained in NVIDIA NeMo)

**Key insight vs. naive cosine + threshold**:
Spectral clustering uses the **global structure** of all pairwise similarities simultaneously, rather than making local greedy decisions. A pair of embeddings with cosine=0.28 (below threshold 0.30) might still be clustered together if they're both strongly connected to other embeddings from the same speaker. The eigengap naturally determines the number of speakers without a manual threshold.

**Failure modes**: Struggles when the number of embeddings is very small. Sensitive to the pruning parameter across domains. Speaker count estimation errors of 1-2.4 speakers observed on challenging datasets (DIHARD-III).

---

## 4. Agglomerative Hierarchical Clustering (AHC)

**Key references**:
- Chen & Gopalakrishnan, "Clustering via the Bayesian Information Criterion" (ICASSP 1998)
- "End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization" (arXiv:2401.12850)

**How it works**:
1. Start with each segment as its own cluster.
2. Compute pairwise distance matrix (cosine distance or PLDA score) between all cluster centroids.
3. Merge the pair with highest similarity (or lowest distance).
4. Update the distance matrix with the new merged cluster.
5. Repeat until stopping criterion is met (typically BIC < 0, or fixed threshold on similarity).

**Online or offline**: Offline. Needs all segments to compute the full distance matrix. However, can be applied to buffered windows.

**How it handles over-segmentation**:
- The stopping criterion determines the final number of clusters. BIC stopping tends to be conservative (better at avoiding over-segmentation than under-segmentation).
- Can be configured to deliberately over-cluster first (creating more clusters than speakers), then do exhaustive search for best cluster combination.
- When combined with PLDA scoring (instead of cosine), the merging decisions account for within-speaker variability.

**Computational cost**:
- O(N^2) distance computations initially, then O(N) updates per merge. Total: O(N^2 * D + N^2) for N segments.
- **Warning**: Can become very slow for long recordings (>30 minutes = thousands of segments). This is a known bottleneck in VBx initialization.

**Needs training data**: Depends on the distance metric. Cosine: no. PLDA: yes. BIC stopping: no (model-based criterion).

**Open-source implementations**: Widely available in scipy, scikit-learn, SpeechBrain, Kaldi.

**Key insight vs. naive cosine + threshold**:
AHC makes globally optimal merge decisions at each step (greedy, but considering all pairs). It naturally produces a hierarchy of solutions from N clusters down to 1, and the stopping criterion selects the right level. Unlike threshold-based assignment, AHC doesn't fragment a speaker just because one segment fell slightly below threshold.

---

## 5. EEND (End-to-End Neural Diarization)

**Key references**:
- Fujita et al., "End-to-End Neural Speaker Diarization with Self-Attention" (2019)
- "LS-EEND: Long-Form Streaming End-to-End Neural Diarization" (arXiv:2410.06670)
- "Online Streaming End-to-End Neural Diarization" (arXiv:2101.08473)
- "O-EENC-SD: Efficient Online End-to-End Neural Clustering" (MERL, 2025)
- GitHub: `hitachi-speech/EEND`, `Audio-WestlakeU/FS-EEND`

**How it works**:
EEND replaces the traditional pipeline (embedding + clustering) with a single neural network that directly outputs frame-level speaker activity labels. The network uses self-attention to model both speaker characteristics and temporal dependencies. Speakers are represented as "attractors" -- learned vectors that attend to their own speech regions.

Streaming variants (LS-EEND, BW-EDA-EEND):
- Process audio in fixed-length blocks.
- Use recurrent mechanisms (retention, RNN) to carry speaker information across blocks.
- Online attractor extraction and update.

**Online or offline**:
- Original EEND: offline (needs full recording).
- LS-EEND: frame-wise streaming with ~1s latency. Handles up to 8 speakers and 1-hour recordings.
- Speaker-Tracing Buffer (STB): streaming with 1s latency, comparable to offline performance.

**How it handles over-segmentation**:
- The model learns from labeled data what constitutes a speaker change vs. within-speaker variation. Over-segmentation patterns are directly penalized during training.
- Attention-based attractors capture global speaker representations, not just local similarities.
- Temporal self-attention inherently smooths predictions over time.

**Computational cost**:
- Self-attention: O(T^2 * D) for T frames, which is expensive for long recordings.
- Streaming variants reduce this to O(T * D) per block using retention/RNN.
- Requires GPU for training; inference can run on CPU for short blocks but is heavier than clustering-based methods.
- Not obviously feasible for <200ms real-time on CPU without optimization.

**Needs training data**: Yes, heavily. Requires fully time-stamped speaker annotations for supervised training. Typically trained on simulated mixtures from single-speaker datasets.

**Open-source implementations**:
- `hitachi-speech/EEND` (PyTorch)
- `Audio-WestlakeU/FS-EEND` (PyTorch, ICASSP 2024)

**Key insight vs. naive cosine + threshold**:
EEND eliminates the clustering step entirely. It directly learns the mapping from audio features to speaker labels, capturing complex patterns (overlapping speech, speaker turns, acoustic similarity) that rule-based clustering cannot. However, it requires significant training data and compute.

**Verdict for VoxTerm**: Too heavy for real-time on-device use. Training data requirements are substantial. Better suited as a post-processing refinement on buffered audio.

---

## 6. Sticky HDP-HMM

**Key references**:
- Fox, Sudderth, Jordan & Willsky, "A sticky HDP-HMM with application to speaker diarization" (Annals of Applied Statistics, 2011, arXiv:0905.2592)

**How it works**:
The Hierarchical Dirichlet Process HMM (HDP-HMM) is a Bayesian nonparametric model that can discover the number of speakers automatically. Each HMM state represents a speaker, and the HDP prior allows an unbounded number of states. However, the basic HDP-HMM over-segments badly because it has no bias toward temporal persistence.

The **sticky** extension adds a self-transition bias parameter (kappa). Each state's transition distribution is augmented with extra probability mass on the self-transition. This makes the model prefer to stay in the current speaker state rather than switch, directly preventing rapid switching and over-segmentation.

Inference uses MCMC sampling with a truncated DP approximation to jointly resample the full state sequence.

**Online or offline**: Offline. MCMC sampling requires multiple passes over the data. Not suitable for streaming.

**How it handles over-segmentation**:
- The kappa (stickiness) parameter is the key mechanism. Higher kappa = stronger preference for staying in the current speaker state.
- Without kappa, the basic HDP-HMM "tends to over-segment the audio data by creating redundant states and rapidly switching among them." The sticky extension was specifically designed to fix this.
- Bayesian nonparametric inference automatically penalizes creating more speakers than the data supports.

**Computational cost**:
- MCMC sampling is iterative and relatively expensive. Not suitable for real-time (<200ms).
- Inference time scales with recording length and number of MCMC iterations.

**Needs training data**: No in the traditional sense -- it's fully Bayesian. But hyperparameters (kappa, concentration parameters) need tuning. Speaker-specific emission models may need prior specification.

**Open-source implementations**: Various research implementations exist; not widely productionized.

**Key insight vs. naive cosine + threshold**:
The sticky HDP-HMM formally models the temporal structure of speaker turns. It knows that speakers persist for extended periods and that switching is rare. This is the fundamental prior that cosine + threshold completely ignores. The same insight (self-transition bias / speaker continuity prior) is what makes VBx's loopP parameter so effective.

---

## 7. UIS-RNN (Unbounded Interleaved-State RNN)

**Key references**:
- Zhang et al., "Fully Supervised Speaker Diarization" (2019, arXiv:1810.04719)
- GitHub: `google/uis-rnn`

**How it works**:
Each speaker is modeled by a parameter-sharing RNN. The RNN states for different speakers interleave in the time domain. A distance-dependent Chinese Restaurant Process (ddCRP) accommodates unknown number of speakers -- the probability of creating a new speaker depends on the distance from existing speaker representations. The system is fully supervised, learning from time-stamped speaker labels.

**Online or offline**: Online. Decodes in an online fashion, assigning each new embedding to either an existing speaker or a new speaker as it arrives. This is one of the few approaches that is natively online.

**How it handles over-segmentation**:
- The ddCRP prior controls the creation of new speakers: a new speaker is only created when the embedding is sufficiently different from all existing speakers.
- The RNN maintains a memory of each speaker's characteristics, providing more robust comparison than a single centroid.
- Temporal modeling is built into the RNN structure.

**Computational cost**:
- RNN forward pass per embedding: O(D * H) where H is hidden size. Fast enough for real-time.
- 7.6% DER on NIST SRE 2000 CALLHOME (better than spectral clustering at the time).

**Needs training data**: Yes, fully supervised. Needs time-stamped speaker labels for training.

**Open-source implementations**: `google/uis-rnn` (Python/PyTorch)

**Key insight vs. naive cosine + threshold**:
UIS-RNN uses a learned distance function (the RNN) rather than raw cosine similarity, and the ddCRP provides a principled way to decide "same speaker or new speaker" that accounts for the number of existing speakers (the more speakers already exist, the higher the bar for creating a new one -- a natural regularizer against over-segmentation).

---

## 8. NME-SC (Auto-Tuning Spectral Clustering)

**Key references**:
- Park & Han, "Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap" (IEEE SPL 2020, arXiv:2003.02405)
- Self-Tuning Spectral Clustering (SC-pNA) (arXiv:2410.00023)
- GitHub: `tango4j/Auto-Tuning-Spectral-Clustering` (now maintained in NVIDIA NeMo)

**How it works**:
NME-SC uses normalized maximum eigengap values to auto-tune two things simultaneously:
1. The number of speakers (from eigengap analysis).
2. The row-wise pruning threshold for the affinity matrix (sweeps over threshold values and picks the one maximizing the eigengap).

SC-pNA (2024 improvement) goes further:
1. For each row of the affinity matrix, apply k-means (k=2) to separate "same-speaker" similarities from "different-speaker" similarities.
2. Keep only the top p% (default 20%) from the same-speaker cluster.
3. Single eigendecomposition to estimate speaker count.
4. K-means on eigenvectors for final assignment.

**Online or offline**: Offline (needs full affinity matrix).

**How it handles over-segmentation**:
- Auto-tuning eliminates the need for a manual threshold, which when set too tight causes over-segmentation.
- The eigengap naturally prefers parsimonious solutions (fewer clusters when evidence is ambiguous).
- SC-pNA's adaptive per-row pruning handles imbalanced speaker distributions better than global thresholds.

**Computational cost**:
- Similar to standard spectral clustering. SC-pNA is faster because it needs only one eigendecomposition vs. NME-SC's sweep over thresholds.
- GPU-accelerated version in NVIDIA NeMo.

**Needs training data**: No. This is a key advantage. No PLDA, no supervised training. Works directly on cosine similarities.

**Open-source implementations**: NVIDIA NeMo (GPU-accelerated), `tango4j/Auto-Tuning-Spectral-Clustering`

**Key insight vs. naive cosine + threshold**:
The threshold is not a hyperparameter -- it's derived from the data itself. NME-SC asks "what threshold gives the cleanest separation into clusters?" and picks that automatically. This adapts to different recordings, speakers, and acoustic conditions.

---

## 9. BIC-Based Clustering

**Key references**:
- Chen & Gopalakrishnan, "Clustering via the Bayesian information criterion with applications in speech recognition" (ICASSP 1998)
- Han & Narayanan, "A robust stopping criterion for AHC in speaker diarization" (Interspeech 2007)

**How it works**:
The Bayesian Information Criterion (BIC) is a model selection criterion that balances model fit against model complexity. In speaker diarization:

Delta-BIC = log-likelihood of single-speaker model - log-likelihood of two-speaker model - lambda * penalty_for_extra_parameters

If Delta-BIC > 0, the two clusters should be merged (single speaker is a better explanation). If Delta-BIC < 0, keep them separate. The penalty term grows with the number of parameters and the log of the sample size, naturally preventing over-fitting (= over-segmentation).

**Online or offline**: Can work incrementally (compute Delta-BIC for pairs as new data arrives), but traditionally used in offline AHC.

**How it handles over-segmentation**:
- The complexity penalty in BIC explicitly penalizes having too many speakers. Creating an extra speaker must be justified by sufficient improvement in log-likelihood.
- However, BIC is known to not be robust to data source variation -- the same BIC threshold doesn't work well across different recording conditions.

**Computational cost**: Computing Delta-BIC requires fitting Gaussian models to each cluster. O(D^2) per pair comparison. Fast.

**Needs training data**: No. BIC is a model-based criterion derived from the data.

**Open-source implementations**: Available in various diarization toolkits (Kaldi, LIUM).

**Key insight vs. naive cosine + threshold**:
BIC provides a principled statistical test for "should these clusters be merged?" rather than an arbitrary threshold. The penalty term automatically adjusts based on the amount of data -- with more observations, you need stronger evidence to justify an extra speaker.

**Limitations**: Not robust across domains. The lambda scaling factor often needs tuning. Alternatives like Bayes Factor have been proposed to address this.

---

## 10. Cluster Purification / Cross-EM Refinement

**Key references**:
- "Improving Speaker Diarization by Cross EM Refinement" (ICASSP 2006)
- "A Cluster Purification Algorithm for Speaker Diarization System" (ISCID 2014)
- "Frame purification for cluster comparison in speaker diarization"
- "Speaker Clustering and Cluster Purification Methods" (IEEE TASLP 2012)

**How it works**:
After initial clustering (which may over-segment), purification post-processes the clusters:

**Cross-EM Refinement**:
1. Split data into two halves (cross-validation style).
2. Train GMM speaker models on each half.
3. Use models from one half to re-label segments in the other half.
4. Iterate until convergence.
This naturally merges over-segmented clusters: if two clusters actually belong to the same speaker, the EM models will converge to the same speaker model and the segments will be reassigned correctly.

**Frame Purification**: Removes non-discriminative frames from clusters before computing inter-cluster distances. Non-speech frames and frames with poor speaker discrimination are eliminated, improving the quality of merge/split decisions.

**Viterbi Resegmentation**: After clustering, train GMMs for each speaker, then use Viterbi decoding to re-label the entire recording. The Viterbi path naturally smooths out isolated mis-assignments by enforcing minimum duration constraints through HMM topology.

**Online or offline**: Offline (iterative refinement over the full recording). However, can be applied to buffered windows periodically.

**How it handles over-segmentation**:
- Cross-EM directly fixes over-segmentation by allowing clusters to be re-merged. Average DER reduction: 22%, up to 56%.
- Viterbi resegmentation improves cluster purity by +5% with minimal coverage impact (+2%).
- Frame purification gives 15.5% relative improvement by removing misleading frames before cluster comparison.

**Computational cost**: EM iterations are O(T * K * D^2) per iteration. Viterbi is O(T * K^2). Both are fast for typical meeting sizes.

**Needs training data**: No (unsupervised EM on the recording itself).

**Key insight vs. naive cosine + threshold**:
These methods recognize that initial clustering is imperfect and provide principled ways to fix mistakes. The key insight is that **clusters should be revisited and potentially merged** after the initial assignment. VoxTerm's current approach makes irreversible decisions -- once split, never merged.

---

## 11. Online Incremental Clustering (Current Approaches)

**Key references**:
- "Online Speaker Diarization of Meetings Guided by Speech Separation" (2024, arXiv:2402.00067)
- "Highly Efficient Real-Time Streaming and Fully On-Device Speaker Diarization" (2022, arXiv:2210.13690)
- "Real-time multilingual speech recognition and speaker diarization" (PMC, 2024)
- "Overlap-aware low-latency online speaker diarization" (2021, arXiv:2109.06483)

**How it works** (representative system):
1. Process audio in 5-second windows with 90% overlap (500ms step).
2. Extract speaker embeddings (ECAPA-TDNN, 192-dim) from each segment.
3. Compute cosine similarity between new embedding and all existing centroids.
4. If max similarity > threshold delta_new: assign to that speaker, update centroid.
5. If max similarity < threshold delta_new: create new speaker cluster.

Key enhancements over naive approaches:
- **Minimum duration gating** (rho_update): embeddings from segments shorter than a threshold do NOT update centroids. This prevents noisy short-segment embeddings from corrupting speaker representations.
- **Cannot-link constraints**: if the local segmentation model says two segments are different speakers, they cannot be merged even if embeddings are similar.
- **Centroid update by accumulation**: cs <- cs + e (average over all assigned embeddings) rather than EMA. This gives equal weight to all observations rather than forgetting early ones.

**Typical thresholds**: delta_new = 0.7 (cosine similarity) for the system using Whisper + ECAPA-TDNN. Note this is much higher than VoxTerm's 0.30.

**Processing time**: 0.13 seconds per 5.11-second buffer (3% of audio duration). Well within real-time.

**Multi-stage clustering** (Google's on-device system):
- Fallback clusterer for short audio.
- Main clusterer for medium-length.
- Pre-clusterer to compress long-form inputs.
- Upper bounds on computational complexity are configurable per device.

**How it handles over-segmentation**:
- Higher similarity thresholds (0.7 instead of 0.30) dramatically reduce false cluster creation.
- Minimum duration gating prevents noisy embeddings from affecting centroids.
- Cannot-link constraints prevent wrong merges without creating extra clusters.
- Some systems use median filtering or HMM smoothing on output labels.

**Key insight vs. naive cosine + threshold**:
The gap between VoxTerm's threshold (0.30) and working systems (0.70) is striking. Additionally, protecting centroids from noisy short-segment embeddings and using accumulative averaging (not EMA) for centroid updates are critical practical differences.

---

## 12. Pyannote Speaker Diarization Pipeline

**Key references**:
- `pyannote/pyannote-audio` (GitHub)
- `pyannote/speaker-diarization-3.1` (Hugging Face)
- Plaquet & Bredin, "pyannote.audio 2.1 speaker diarization pipeline" (2023)

**How it works**:
Multi-stage pipeline:
1. **Segmentation model**: neural network that produces frame-level speaker activity probabilities for local windows (5-10 seconds).
2. **Embedding extraction**: speaker embeddings extracted from segments identified by the segmentation model.
3. **Clustering**: agglomerative hierarchical clustering with a learned similarity threshold.
4. **Resegmentation**: final pass to refine boundaries.

**Key hyperparameters**:
- `segmentation.threshold`: activation threshold for the segmentation model.
- `clustering.method`: clustering algorithm (default: AHC).
- `clustering.threshold`: similarity threshold for stopping AHC.
- `clustering.min_cluster_size`: minimum number of segments per cluster.
- `min_speakers`, `max_speakers`, `num_speakers`: optional constraints.

**Optimization**: threshold parameters are tuned sequentially -- segmentation threshold first (with oracle clustering), then clustering threshold (with optimal segmentation). This principled tuning avoids over-segmentation from poorly set thresholds.

**Key insight vs. naive cosine + threshold**:
Pyannote separates segmentation from clustering and tunes each independently. Providing num_speakers or min/max speakers constraints can eliminate over-segmentation when speaker count is known or bounded.

---

## 13. Cross-Cutting Concerns

### 13.1 How to Merge Speaker Clusters Online

Techniques for detecting and merging over-segmented clusters in an online setting:

1. **Periodic pairwise re-evaluation**: Every N segments, compute pairwise similarities between all existing centroids. If any pair exceeds a merge threshold, combine them. This catches splits that happened due to noisy early embeddings.

2. **Centroid drift monitoring**: Track how centroids evolve. If two centroids converge over time (their similarity increases as more data is added), merge them.

3. **Count-based regularization**: Inspired by ddCRP in UIS-RNN: the more clusters you already have, the higher the bar for creating a new one. P(new_speaker) ~ alpha / (alpha + N_existing).

4. **Two-threshold approach**: Use a strict threshold for creating new clusters (high confidence required) and a looser threshold for merging existing clusters. This asymmetry biases toward fewer clusters.

5. **Hierarchical online clustering**: Maintain a hierarchy. When a new embedding arrives, assign it to the leaf cluster. Periodically re-evaluate whether leaf clusters should be merged at a higher level.

6. **VBx-on-buffer**: Periodically run VBx or spectral clustering on a rolling buffer of recent embeddings to get a "ground truth" clustering, then map online clusters to this reference.

### 13.2 Embedding Quality Estimation

How to score whether an embedding is reliable before using it for clustering:

1. **Duration-based quality**: Embeddings from segments < 1.5 seconds are significantly less reliable. A simple duration threshold is the most effective quality gate. Recommended minimum: 1.5-2.0 seconds for ECAPA-TDNN.

2. **Uncertainty quantification**: xi-vector networks can output both an embedding and an uncertainty estimate. High uncertainty = unreliable embedding. This uncertainty can be propagated to PLDA scoring.

3. **Energy/SNR gating**: Low-energy segments or segments with poor SNR produce unreliable embeddings. Compute segment SNR and down-weight or discard low-SNR embeddings.

4. **Overlap detection**: Segments containing overlapping speakers produce poor embeddings (the embedding is a mixture of two speakers). Use an overlap detection model to identify and down-weight these segments.

5. **Centroid distance consistency**: If a new embedding's distance to its assigned centroid is much larger than the average distance of existing members, it may be an outlier. Weight it less in centroid updates.

6. **Quality-based score calibration**: Adding speech duration as a quality metric factor (QMF) during score calibration improves EER by 6% and MinDCF by 2%.

**Practical recommendation**: At minimum, gate on segment duration (>1.5s) and energy (>threshold dB). This alone eliminates many noisy embeddings that cause over-segmentation.

### 13.3 Speaker Continuity Prior

Using temporal context in clustering decisions:

1. **HMM transition matrix (VBx-style)**: Model speaker assignments as an HMM where self-transition probability is high (0.95-0.99). This is the most principled approach. Equivalent to saying "the speaker changes with probability 1-5% at each segment boundary."

2. **Sticky HDP-HMM**: Same idea in a Bayesian nonparametric framework. The kappa parameter adds extra probability mass to self-transitions.

3. **Label smoothing / minimum duration constraint**: After clustering, apply a smoothing filter. If a segment is labeled differently from its neighbors, re-assign it if the evidence is weak. Common approaches:
   - Median filter on label sequence.
   - Minimum duration constraint: speaker segments shorter than X seconds are re-assigned to the surrounding speaker.
   - Gaussian kernel smoothing of log-likelihoods (12% improvement in speaker error rate).

4. **Temporal decay in similarity**: Weight recent embeddings more heavily when comparing to centroids, but don't forget old ones. This is different from EMA (which forgets) -- it's a weighted average that still considers all observations.

5. **Turn-level assignment**: Instead of assigning individual segments, assign entire speaker turns (continuous speech from one speaker). This groups correlated segments and prevents isolated mis-assignments.

### 13.4 Minimum Segment Duration and Embedding Quality

Research consistently shows:
- Embeddings need **2-4 seconds** of speech for reliable speaker representation.
- Segments < 1.5 seconds produce noisy embeddings that are the primary cause of over-segmentation.
- Trade-off: longer segments = better embeddings but coarser temporal resolution.
- Solution: use overlapping windows (e.g., 3-second windows with 1.5s overlap) to get both reasonable duration and temporal resolution.

### 13.5 Cosine Similarity vs. PLDA: When to Use Which

| Aspect | Cosine | PLDA |
|--------|--------|------|
| Training data needed | No | Yes (speaker-labeled embeddings) |
| Multi-enrollment | Poor calibration | Well-calibrated |
| Domain mismatch | Degrades | More robust |
| With modern embeddings (ECAPA-TDNN) | Competitive | ~10% better EER |
| Computational cost | O(D) | O(D^2) |
| Implementation complexity | Trivial | Moderate |
| Key advantage | Simplicity | Models embedding noise |

**Practical recommendation**: Start with cosine similarity but with a much higher threshold (0.55-0.70 for ECAPA-TDNN, not 0.30). If over-segmentation persists, add a constrained PLDA scorer -- it's a drop-in replacement that better handles embedding variability.

---

## 14. Recommendations for VoxTerm

Based on this research, here are concrete approaches ordered by implementation complexity and expected impact.

### Tier 1: Quick Wins (high impact, low effort)

1. **Raise the cosine similarity threshold from 0.30 to 0.55-0.65**. Working online systems use 0.70. VoxTerm's 0.30 is far too low, meaning almost everything looks like a new speaker. This alone may cut detected speakers from 5+ to 2-3.

2. **Gate embeddings on segment duration**. Do not use embeddings from segments shorter than 1.5 seconds for clustering decisions. Either discard them or assign them to the most recent speaker (continuity prior). This eliminates the noisiest embeddings.

3. **Replace EMA centroid update with cumulative averaging**. Instead of `centroid = alpha * centroid + (1-alpha) * new_embedding`, use `centroid = mean(all_assigned_embeddings)` or at minimum a much lower alpha (0.5-0.7 instead of 0.95). EMA with alpha=0.95 means new observations barely affect the centroid, so it's essentially frozen after the first few segments -- which means if the first embedding was noisy, the centroid is permanently wrong.

4. **Add minimum-duration label smoothing**. After assignment, apply a rule: if a speaker segment is shorter than 2 seconds and is surrounded by the same other speaker, re-assign it. This catches isolated mis-assignments.

### Tier 2: Moderate Effort, High Impact

5. **Periodic cluster merging**. Every 30 seconds (or every 20 segments), compute pairwise cosine similarity between all centroids. If any pair exceeds 0.60, merge them. This directly fixes the "once split, never merged" problem.

6. **Speaker continuity prior via transition probability**. When deciding whether a new segment is the same or different speaker, multiply the cosine similarity by a continuity bonus (e.g., 1.1x if the previous segment was assigned to this speaker). This encodes the prior that speakers tend to continue. Equivalent to a simplified VBx loopP.

7. **Two-threshold system**. Use a strict threshold for creating NEW speakers (e.g., cosine < 0.40 to ALL existing centroids) but a loose threshold for assigning to EXISTING speakers (e.g., cosine > 0.55 to any centroid). This asymmetry makes it hard to create new speakers but easy to assign to existing ones.

8. **Embedding quality weighting**. Weight centroid updates by segment duration and energy. A 4-second high-energy segment should contribute more to the centroid than a 0.5-second quiet one.

### Tier 3: Significant Effort, Maximum Impact

9. **Spectral clustering on rolling buffer**. Every 60 seconds, run spectral clustering (using Google's `SpectralCluster` library) on all embeddings in the last 2-3 minutes. Use the eigengap-determined speaker count as ground truth and remap online clusters to match. This gives the benefits of global clustering while maintaining online operation.

10. **VBx on rolling buffer**. Same as above but using VBx with loopP=0.99 and Fb=17. More accurate than spectral clustering, handles temporal continuity natively, but requires a pre-trained PLDA model.

11. **Constrained PLDA scoring**. Replace cosine similarity with the "extremely constrained PLDA" from the ICASSP 2023 paper. This handles multi-enrollment (comparing against a centroid built from multiple embeddings) with proper uncertainty modeling. Use a pre-trained PLDA from SpeechBrain.

### Tier 4: Research-Level

12. **UIS-RNN**: Retrain for online use. Needs supervised data but provides native online decoding with ddCRP-based speaker count estimation.

13. **EEND post-processing**: Run a lightweight EEND model on buffered audio to refine cluster assignments. The "End-to-End Speaker Diarization as Post-Processing" approach uses EEND to fix clustering mistakes.

### Summary: Most Important Changes

The three changes most likely to fix VoxTerm's 2-speakers-become-5 problem:

1. **Raise threshold to 0.55-0.65** (instant fix for most over-segmentation)
2. **Add periodic cluster merging** (fixes the "never merge back" structural problem)
3. **Gate on segment duration >= 1.5s** (eliminates noisy embeddings that cause spurious clusters)

These three changes require minimal code and no new dependencies. They address the three root causes identified in the problem statement: static threshold (too low), no merging, and embedding noise from short segments.

---

## Sources

### Papers
- [Speaker diarization with PLDA i-vector scoring](https://ieeexplore.ieee.org/document/7078610/)
- [Probabilistic Back-ends for Online Speaker Recognition](https://arxiv.org/abs/2302.09523)
- [Bayesian HMM clustering of x-vector sequences (VBx)](https://arxiv.org/abs/2012.14952)
- [Discriminative Training of VBx](https://arxiv.org/abs/2310.02732)
- [Speaker Diarization with LSTM (Google)](https://google.github.io/speaker-id/publications/LstmDiarization/)
- [Auto-Tuning Spectral Clustering (NME-SC)](https://arxiv.org/abs/2003.02405)
- [Self-Tuning Spectral Clustering (SC-pNA)](https://arxiv.org/abs/2410.00023)
- [Robustness of Spectral Clustering for Diarization](https://arxiv.org/abs/2403.14286)
- [Scoring of Large-Margin Embeddings: Cosine or PLDA?](https://arxiv.org/abs/2204.03965)
- [Unifying Cosine and PLDA Back-ends](https://arxiv.org/abs/2204.10523)
- [A Sticky HDP-HMM with Application to Speaker Diarization](https://arxiv.org/abs/0905.2592)
- [Fully Supervised Speaker Diarization (UIS-RNN)](https://arxiv.org/abs/1810.04719)
- [End-to-End Neural Diarization (EEND)](https://github.com/hitachi-speech/EEND)
- [LS-EEND: Long-Form Streaming EEND](https://arxiv.org/abs/2410.06670)
- [Online Streaming End-to-End Neural Diarization](https://arxiv.org/abs/2101.08473)
- [Efficient Online End-to-End Neural Clustering (O-EENC-SD)](https://www.merl.com/publications/docs/TR2025-031.pdf)
- [Highly Efficient Real-Time On-Device Diarization](https://arxiv.org/abs/2210.13690)
- [Online Speaker Diarization of Meetings](https://arxiv.org/abs/2402.00067)
- [Overlap-aware Low-Latency Online Diarization](https://arxiv.org/abs/2109.06483)
- [ECAPA-TDNN Embeddings for Speaker Diarization](https://arxiv.org/abs/2104.01466)
- [End-to-End Speaker Diarization as Post-Processing](https://arxiv.org/abs/2012.10055)
- [DiarizationLM: Post-Processing with LLMs](https://arxiv.org/abs/2401.03506)
- [DOVER-Lap: Combining Diarization Outputs](https://arxiv.org/abs/2011.01997)
- [Real-time multilingual speaker diarization](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041969/)
- [Incorporating Uncertainty from Speaker Embedding Estimation](https://ieeexplore.ieee.org/document/10097019/)
- [Transition Cost and Model Parameters in Diarization](https://link.springer.com/article/10.1186/s13636-021-00196-6)
- [A Review of Speaker Diarization](https://www.preprints.org/manuscript/202412.2368)
- [Cross EM Refinement for Speaker Diarization](https://ieeexplore.ieee.org/document/4036996/)
- [Cluster Purification Algorithm](https://ieeexplore.ieee.org/document/7082048/)
- [BIC Clustering (Chen & Gopalakrishnan 1998)](https://ieeexplore.ieee.org/document/675347/)
- [Robust BIC Stopping Criterion](https://www.researchgate.net/publication/221479915/)
- [VBx Implementation Notes](https://desh2608.github.io/2022-04-10-gbo-vb/)
- [Similarity Measurement of Speaker Embeddings in Diarization](https://sites.duke.edu/dkusmiip/files/2022/11/Similarity-Measurement-of-Segment-level-Speaker-Embeddings-in-Speaker-Diarization.pdf)

### Repositories
- [BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx) - Variational Bayes HMM diarization
- [wq2012/SpectralCluster](https://github.com/wq2012/SpectralCluster) - Google's spectral clustering
- [tango4j/Auto-Tuning-Spectral-Clustering](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering) - NME-SC
- [google/uis-rnn](https://github.com/google/uis-rnn) - UIS-RNN
- [hitachi-speech/EEND](https://github.com/hitachi-speech/EEND) - End-to-end neural diarization
- [Audio-WestlakeU/FS-EEND](https://github.com/Audio-WestlakeU/FS-EEND) - Frame-wise streaming EEND
- [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) - Pyannote diarization toolkit
- [prachiisc/PLDA_scoring](https://github.com/prachiisc/PLDA_scoring) - PLDA scoring implementation
- [SpeechBrain PLDA module](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.PLDA_LDA.html)
- [desh2608/dover-lap](https://github.com/desh2608/dover-lap) - DOVER-Lap system combination
- [NVIDIA NeMo NME-SC](https://github.com/NVIDIA/NeMo) - GPU-accelerated spectral clustering

# Research Context for Copilot
## Project: Robust Deep Multimodal Audio-Visual Steganography / Audio-First Extension

### IMPORTANT INSTRUCTION FOR COPILOT

Treat this document as the current research context and architectural direction. Do **not** immediately implement the largest proposed architecture. The goal is to move from a known baseline to measurable improvements through small, testable experiments.

Use the terminology below consistently:

- `cover` = innocent host media
- `secret` = hidden media
- `stego` = cover after secret embedding
- `r` = true auxiliary/redundancy latent produced by an INN during forward encoding
- `z` = latent supplied to the inverse INN at decoding; in prior baselines this may be sampled from Gaussian noise
- `r_hat` / `z_hat` = predicted auxiliary/redundancy latent
- `y` = clean stego audio
- `y'` = attacked/corrupted stego audio

Do not call the INN redundancy latent a "random seed" in technical writing. "Auxiliary latent" or "redundancy latent" is more precise.

---

# 1. Original Project Problem

The original capstone problem is:

> Design a deep-learning-based multimodal steganography framework that can jointly embed and extract a high-capacity secret payload, especially a full video including its imagery and audio, inside a host/cover video while maintaining imperceptibility in each modality, time synchronization, and robustness to common distortions such as compression.

The intended system is:

```text
                    COVER A/V
                   /         \
             Cover Video    Cover Audio

                    +

                    SECRET A/V
                   /          \
             Secret Video   Secret Audio

                    |
                    v
            Temporal Synchronization
                    |
             fixed aligned chunks
                  /     \
                 /       \
          Video-INN     Audio-INN
               |            |
          Stego video   Stego audio
                 \        /
                  \      /
                  MUX / container
                       |
                       v
                  STEGO A/V
```

At decoding:

```text
STEGO A/V
   |
   v
demultiplex + synchronization
   |
   +-------------------+
   |                   |
   v                   v
Video decoder      Audio decoder
   |                   |
   v                   v
Secret video       Secret audio
```

The fundamental objectives are:

```text
Stego video ≈ Cover video
Stego audio  ≈ Cover audio

Recovered secret video ≈ Original secret video
Recovered secret audio  ≈ Original secret audio

Recovered audio and video remain synchronized.
```

---

# 2. What Already Exists in the Literature

## 2.1 Traditional / neural steganography

Traditional steganography used fixed rules such as LSB modification. Deep learning replaced fixed rules with learned embedding/recovery mappings.

The important trend for this project is the use of **Invertible Neural Networks (INNs)** because the transformation can be analytically reversed.

---

## 2.2 Hide and Speak

Hide and Speak is a deep speech steganography approach.

Conceptually:

```text
carrier speech + secret speech
          |
       encoder
          |
      stego audio
          |
     message decoder
          |
     recovered secret
```

Important ideas:
- differentiable STFT/iSTFT during training;
- waveform + spectral objectives;
- supports hiding multiple secret messages;
- evaluates intelligibility using measures such as WER/CER and perceptual tests;
- reports robustness to distortions such as compression/noise/downsampling/filtering.

Limitations relative to this project:
- not a full A/V synchronized framework;
- not based on the clean INN reversible design we want;
- previous review noted sensitivity to quantization/compression and quality trade-offs.

Lesson:
- audio fidelity must be considered in both time and frequency domains;
- audio steganography cannot be evaluated only by waveform loss.

---

## 2.3 LF-VSN

LF-VSN is the main video foundation.

Core idea:

```text
cover video + secret video
          |
         INN
          |
     stego video + auxiliary/redundancy latent
```

It uses:
- INN-based reversible hiding;
- temporal windows/context;
- frequency/wavelet feature processing;
- additive + affine coupling blocks;
- redundancy prediction;
- reverse INN decoding.

The important LF-VSN trick is:

```text
If the true redundancy latent is not transmitted,
do not simply sample random Gaussian redundancy.

Instead:

stego/context
     |
     v
Redundancy Prediction Module (RPM)
     |
     v
predicted redundancy r_hat
     |
     v
inverse INN
     |
     v
recovered secret
```

This showed that content-dependent redundancy prediction can improve stable recovery versus purely random auxiliary sampling.

Limitations:
- video-only;
- does not provide the full synchronized audio-video framework desired here;
- real-world compression/distortion can threaten exact invertibility.

Lesson:
- predicting discarded INN redundancy is useful;
- but prediction of redundancy alone does not automatically make the whole system robust if the stego itself is corrupted.

---

## 2.4 WavInWav

WavInWav is the main audio-INN baseline.

Conceptually:

```text
cover audio c
+
secret audio s
      |
     INN
      |
      +--------> stego audio y
      |
      +--------> auxiliary/redundancy r
```

At exact reverse time:

```text
(y, r)
   |
 INN^-1
   |
(c, s)
```

But the practical system does not transmit the true `r`. The auxiliary variable may instead be replaced/sampled with Gaussian noise:

```text
stego y
   +
random z ~ N(0,I)
       |
    INN^-1
       |
secret estimate
```

WavInWav also uses fixed-rate/chunked audio processing and time-frequency reconstruction losses.

Important lesson:
- full secret audio can be hidden using an INN;
- the auxiliary/redundancy latent is a key part of practical decoding;
- Gaussian replacement creates an opportunity for learned redundancy prediction.

---

# 3. What Our Team Has Actually Implemented So Far

According to the current review:

### Video side is substantially implemented

Existing work includes:
- Vimeo90K data pipeline;
- cover-secret video pairing;
- preprocessing;
- 3-frame temporal windows;
- Haar DWT/IWT;
- Video-INN;
- additive/affine coupling blocks;
- latent prediction module;
- reverse Video-INN decoding;
- quantization / 8-bit simulation;
- testing;
- PSNR/SSIM evaluation.

The current review explicitly states that **Audio-INN training and complete A/V synchronization were still pending** at that point.

Therefore, the real current baseline is closer to:

```text
Cover video + Secret video
          |
      Video-INN
          |
      Stego video
          |
    latent prediction
          |
      inverse INN
          |
    recovered secret
```

The full:

```text
Video-INN + Audio-INN + synchronized A/V system
```

is the intended original system, not something that should be described as already completely finished.

---

# 4. Main Weaknesses of the Original INN Architecture

## 4.1 "Reversible" does not automatically mean "blindly recoverable"

The ideal INN is:

```text
(Cover, Secret)
       |
      INN
       |
(Stego, r)
```

The exact inverse needs:

```text
(Stego, r)
       |
    INN^-1
       |
(Cover, Secret)
```

But if only `Stego` is transmitted and `r` is discarded:

```text
Stego
  |
predict/sample r
  |
INN^-1
  |
Secret_hat
```

then the practical system is not exact end-to-end inversion from the stego alone.

This is the main conceptual issue with claiming "perfect reversibility" for the deployed system.

---

## 4.2 Compression breaks the exact inverse assumptions

Let:

```text
clean stego = y
```

After MP3/AAC/resampling/filtering/noise:

```text
attacked stego = y' = A(y)
```

and generally:

```text
y' != y
```

Even if the true redundancy `r` were known:

```text
INN^-1(y', r)
```

is not the same operation as:

```text
INN^-1(y, r)
```

Therefore robust recovery has two possible failure sources:

1. missing/wrong auxiliary latent;
2. corrupted stego signal itself.

This distinction must be experimentally isolated.

---

## 4.3 Current audio/video branches are synchronized but not deeply multimodal

Original architecture:

```text
Video-INN + Audio-INN + common time/chunking
```

means the system is unified at the framework/timeline level, but the neural branches are largely independent.

A stronger future multimodal architecture would allow one modality to help recover the other.

---

## 4.4 Synchronization is currently based mainly on shared chunk indexing

Example:

```text
chunk 0:
video 0-1 sec <-> audio 0-1 sec

chunk 1:
video 1-2 sec <-> audio 1-2 sec
```

This prevents ordinary drift, but real attacks such as trimming/dropping/re-timing can break naive indexing.

Future versions may need explicit synchronization pilots/counters or content-based re-alignment.

---

## 4.5 Evaluation is currently too focused on clean quality

Current video evaluation emphasizes:
- PSNR;
- SSIM;
- cover reconstruction quality;
- secret reconstruction quality.

For robustness research, additionally evaluate:
- MP3/AAC/Opus;
- resampling;
- quantization;
- additive noise;
- filtering;
- time shifts/cropping where appropriate;
- actual codec outputs;
- intelligibility and content preservation.

---

# 5. Our Current Research Strategy

There are two possible research directions.

## Direction B — FIRST PRIORITY

### Focus on robust audio steganography

Use the existing WavInWav-style audio-INN baseline and improve the practical decoding/recovery path.

Core idea:

```text
WavInWav
+
LF-VSN-style redundancy prediction
+
attack-aware prediction
+
conditional latent diffusion
+
optional stego correction
```

This is the first direction to implement because:
- it can reuse a known audio-INN baseline;
- it does not require rebuilding the video model;
- improvements can be made incrementally;
- each stage yields a clear ablation/result;
- it is easier to debug than a full AV Transformer.

---

# 6. Direction B: The Key Research Hypothesis

The most basic hypothesis is:

> Instead of supplying random Gaussian auxiliary noise to the inverse INN, can we infer the auxiliary/redundancy latent from the observed stego audio, even after the stego has been distorted?

Baseline:

```text
attacked stego y'
       +
random Gaussian z
       |
    INN^-1
       |
secret_hat
```

Proposed first improvement:

```text
attacked stego y'
       |
       v
Audio Redundancy Prediction Module (RPM)
       |
       v
r_hat
       |
       v
INN^-1(y', r_hat)
       |
       v
secret_hat
```

This direct RPM idea is inspired by LF-VSN and is **not itself novel**. It should be treated as the first baseline/improvement, not the final contribution.

---

# 7. The Main New Idea: Conditional Diffusion for Redundancy Recovery

Instead of predicting `r` once through a deterministic network:

```text
y' -> CNN -> r_hat
```

learn a conditional distribution over possible redundancy states:

```text
p(r | attacked stego)
```

Use a diffusion model to progressively recover the redundancy:

```text
attacked stego
      |
condition encoder
      |
      v
conditioning features c
      |
      +
partially/noisy redundancy state r_t
      |
      v
conditional diffusion
      |
      v
refined redundancy r_hat
```

The diffusion model should not directly denoise the stego waveform first.

Avoid:

```text
attacked stego
   |
diffusion denoiser
   |
secret extractor
```

because steganographic information can itself look like a tiny perturbation.

Prefer:

```text
attacked stego
   |
robust condition / extractor
   |
corrupted secret-related representation
   |
conditional latent diffusion
   |
recovered INN redundancy
   |
inverse INN
   |
secret
```

---

# 8. StyleGAN / AdaIN Inspiration

The conceptual analogy is:

### StyleGAN

```text
style latent
   |
   +--> modulate early layers
   +--> modulate middle layers
   +--> modulate late layers
   |
progressive image synthesis
```

### Proposed audio system

```text
stego-derived conditioning
   |
   +--> modulate early diffusion blocks
   +--> modulate middle diffusion blocks
   +--> modulate late diffusion blocks
   |
progressive redundancy reconstruction
```

The idea is that the information extracted from the stego should act as a persistent condition throughout the diffusion process.

Do not necessarily use literal AdaIN because normalization can discard useful information and is awkward with strict invertibility.

Prefer:
- FiLM;
- adaptive GroupNorm;
- feature-wise affine modulation;
- cross-attention where useful.

Example:

```text
h' = gamma(c,t) * GN(h) + beta(c,t)
```

where:
- `h` = diffusion feature;
- `c` = stego-derived condition;
- `t` = diffusion timestep;
- `gamma`, `beta` = learned modulation parameters.

This is "StyleGAN-like" in concept, not a literal copy.

---

# 9. Very Important: Two Different Types of "Noise"

Do not conflate these.

## Diffusion noise

Artificial Gaussian noise added to the redundancy latent:

```text
r
 |
add diffusion noise at timestep t
 |
r_t
```

The model learns to predict/remove this noise.

## Channel corruption

Real transformations applied to the stego:

```text
y
 |
MP3
AAC
Opus
resampling
filtering
quantization
Gaussian noise
time changes
 |
y'
```

These are not equivalent to Gaussian diffusion noise.

The desired model should ultimately handle both:

```text
diffusion timestep t
+
channel attack type/severity a
```

Conceptually:

```text
D(r_t, t, Encoder(A_a(y)))
        |
        v
      r_hat
```

---

# 10. Critical Oracle Experiment Before Building Diffusion

This should be the **first experiment**.

Use a trained Audio-INN and compare:

| Case | Stego | Auxiliary latent |
|---|---|---|
| A | clean `y` | true `r` |
| B | clean `y` | random Gaussian `z` |
| C | attacked `y'` | true `r` |
| D | attacked `y'` | random Gaussian `z` |

Interpretation:

### If A excellent, B worse, C still reasonably good, D poor:

Main bottleneck = missing redundancy.

Proceed strongly with RPM + diffusion.

### If A excellent, B good, C terrible, D terrible:

Main bottleneck = corruption of the stego itself.

Predicting redundancy alone cannot solve the problem. We need a stego-repair/correction branch too.

This experiment prevents wasting time on the wrong subproblem.

---

# 11. Proposed Full Direction-B Architecture

Initial version:

```text
            COVER AUDIO c
                 +
            SECRET AUDIO s
                 |
                 v
               Audio-INN
                 |
            +----+----+
            |         |
            v         v
         stego y      r
                      |
               training target
            (not transmitted)

                   CHANNEL
                     |
                     v
              attacked stego y'
                     |
                     v
             Condition Encoder
                     |
                     v
              multi-scale c
                     |
          +----------+----------+
          |                     |
          v                     v
   optional stego repair    latent diffusion
          |                     |
          v                     v
      y_corr              r_hat / z_hat
          |                     |
          +----------+----------+
                     |
                     v
            Audio-INN inverse
                     |
                 +---+---+
                 |       |
                 v       v
             cover_hat secret_hat
```

Important:
- keep the INN largely frozen initially;
- train the new recovery components first;
- only fine-tune the INN after the new modules prove useful.

---

# 12. Predict-Then-Diffuse Variant

A particularly practical version is:

```text
attacked stego y'
       |
       v
small deterministic RPM
       |
       v
coarse redundancy r_coarse
       |
       v
add controlled diffusion noise
       |
       v
conditional latent diffusion
       |
       v
refined redundancy r_hat
       |
       v
INN inverse
       |
       v
secret_hat
```

This is preferred as the first diffusion experiment because it:
- reduces the diffusion search space;
- gives a strong deterministic baseline;
- can use fewer diffusion steps;
- makes ablations very clean.

Compare:

```text
random z
vs
RPM
vs
diffusion from Gaussian
vs
RPM + diffusion
```

---

# 13. Optional Second Problem: Repair the Corrupted Stego

If the oracle experiment shows that `y'` itself is the major issue, introduce:

```text
y' -> Stego Repair Network -> y_corr
```

where:

```text
y_corr = y' + Delta_hat
```

Then:

```text
INN^-1(y_corr, r_hat)
```

The deterministic correction module should come before trying joint diffusion over stego correction and redundancy.

If successful, a later version can model both:

```text
[r, Delta]
```

with one conditional diffusion process.

Important:
- `Delta` is a general channel-distortion residual;
- do not call it simply "the noise" because codec compression and resampling are nonlinear/non-additive.

---

# 14. Diffusion Training

For a training sample:

```text
F(c,s) = (y,r)
```

Apply a real or simulated attack:

```text
y_a = A(y)
```

Sample a random diffusion timestep:

```text
t ~ Uniform(1,T)
```

Create:

```text
r_t = sqrt(alpha_bar_t) * r
      +
      sqrt(1-alpha_bar_t) * epsilon
```

Conditional diffusion predicts:

```text
epsilon_hat = D(r_t, t, Encoder(y_a))
```

Recover an estimate of clean redundancy:

```text
r_hat_0 =
    [r_t - sqrt(1-alpha_bar_t) * epsilon_hat]
    /
    sqrt(alpha_bar_t)
```

Then pass through the inverse INN:

```text
secret_hat = INN^-1(y_corr, r_hat_0)
```

---

# 15. Training Losses

Candidate total loss:

```text
L =
    lambda_diff   * L_diffusion
  + lambda_r      * L_redundancy
  + lambda_repair * L_stego_repair
  + lambda_secret * L_secret
```

where:

### Diffusion loss

```text
L_diffusion = ||epsilon_hat - epsilon||^2
```

### Redundancy loss

```text
L_redundancy = distance(r_hat, r)
```

### Stego repair loss

```text
L_stego_repair = distance(y_corr, y)
```

### Secret reconstruction loss

Use multiple audio-domain objectives:

```text
waveform L1/L2
+
multi-resolution STFT loss
+
SI-SDR where appropriate
```

The downstream inverse-INN secret loss is important because the best `r_hat` for the research task is not necessarily the one with minimum latent MSE; what matters is whether it allows useful secret reconstruction.

---

# 16. Potential Deeper Formulation

Do not assume the goal is necessarily:

```text
r_hat == exact r
```

There may be multiple auxiliary states that produce acceptable inversion.

The more useful downstream goal may be:

```text
find z_hat such that

INN^-1(y_corr, z_hat) -> correct secret
```

Therefore a later formulation could optimize the conditional posterior of **useful compensation latents**, rather than only the exact discarded latent.

Initially supervise with the known `r` because it is available during training.

Later investigate whether inverse-INN secret reconstruction is a better ultimate supervision signal.

---

# 17. Existing Research and What We Learn From It

Important existing directions:

### WavInWav
Already does:
- INN audio-in-audio hiding;
- auxiliary/redundancy variable;
- Gaussian replacement;
- noise/resampling robustness training.

Therefore:
> "Use an INN for audio steganography" is not new.

### LF-VSN
Already does:
- INN video hiding;
- temporal context;
- wavelet/frequency processing;
- redundancy prediction rather than random Gaussian sampling.

Therefore:
> "Predict the INN redundancy" is not new.

### Diffusion audio steganography
Recent work already uses diffusion directly for audio steganography, including robust message hiding.

Therefore:
> "Use diffusion for audio steganography" is not sufficient novelty.

### Diffusion audio restoration/enhancement
Diffusion is already well established for degraded/noisy speech/audio reconstruction.

Therefore:
> "Use diffusion to restore corrupted audio" is not sufficient novelty.

### Stronger gap
The interesting intersection is:

```text
INN audio steganography
+
missing auxiliary/redundancy recovery
+
attack-aware conditioning
+
conditional latent diffusion
+
inverse-INN downstream recovery
```

Potential contribution:

> A stego-conditioned diffusion framework that progressively infers the missing auxiliary state required by an invertible audio steganography decoder under channel corruption, with reconstruction supervision through the inverse INN.

This exact combination should be presented cautiously as a research gap, not as an absolute claim that no one has ever done it.

---

# 18. Why Direction B Comes First

Direction B should be the immediate research priority because:

```text
existing Audio-INN baseline
          |
          v
small RPM
          |
          v
attack-aware RPM
          |
          v
conditional diffusion
          |
          v
optional stego repair
```

Each change is incremental.

It also allows quick experimental milestones.

The goal is:

> "small change -> measurable improvement -> understand why -> next change"

rather than building an enormous new architecture all at once.

---

# 19. Experimental Plan for Direction B

## Baselines

1. Random Gaussian auxiliary latent.
2. True auxiliary latent oracle.
3. Deterministic RPM.
4. Attack-trained RPM.
5. Diffusion initialized from Gaussian.
6. RPM + diffusion.
7. RPM + diffusion + stego correction if necessary.

## Attacks

Start simple:
- identity / clean;
- Gaussian noise;
- resampling;
- quantization.

Then add real codecs:
- MP3;
- AAC;
- Opus.

Also test severity ranges:
- multiple bitrates;
- multiple SNRs;
- multiple resampling conditions.

Prefer actual codec-generated data for final evaluation, even if training first uses differentiable approximations.

## Metrics

### Cover/stego fidelity
- waveform distortion;
- multi-resolution STFT distance;
- SI-SDR or equivalent where useful.

### Secret recovery
- waveform reconstruction;
- STFT quality;
- SI-SDR;
- STOI;
- PESQ where appropriate;
- WER/content accuracy using ASR;
- speaker similarity if relevant.

### Latent recovery
- latent MSE/cosine distance;
- downstream inverse-INN secret quality.

### Robustness
Plot recovery quality against attack severity.

Especially useful:

```text
0% latent corruption
10%
25%
50%
75%
```

and compare methods.

---

# 20. What "success" looks like

The ideal behavior is graceful degradation:

```text
CLEAN
steo -> actual/full latent recovery -> high-quality secret

MILD ATTACK
steo' -> mostly correct redundancy -> near-clean secret

MODERATE ATTACK
steo' -> partial recovery -> diffusion reconstruction -> good secret

STRONG ATTACK
steo' -> only robust information survives -> lower-fidelity but still intelligible secret

EXTREME ATTACK
critical information lost -> reconstruction becomes unreliable
```

Important:
- perceptually plausible output is not enough;
- if the model changes spoken words, secret recovery is not truly correct;
- distinguish exact/transmitted information from generated perceptual detail.

A useful principle:

> Transmit critical information; generate/reconstruct fidelity.

---

# 21. Direction A — FUTURE Extension After Direction B

Only attempt Direction A if Direction B gives useful results and there is sufficient project time.

## Goal

Turn the system into a genuinely multimodal neural architecture rather than:

```text
Video-INN + Audio-INN + synchronization
```

The idea is:

```text
Cover Video -> Video features ----\
                                    \
                                     -> AV Transformer
                                    /       |
Cover Audio -> Audio features ----/        |
                                             v
                                   shared multimodal context
                                      /              \
                                     /                \
                              Audio head          Video head
                                 |                    |
                              rA_hat                rV_hat
```

The Transformer can model:
- temporal alignment;
- cross-modal context;
- where embedding is safer / less perceptible;
- cross-modal redundancy prediction.

---

# 22. Important Correction to "Do Not Change the Original Data"

It is impossible to hide new information while literally leaving the carrier unchanged.

Correct goal:

```text
stego ~= cover
```

where perturbations are:
- small;
- perceptually masked;
- content adaptive.

Attention can learn where/how strongly to perturb, but cannot encode information with zero modification.

---

# 23. Why a ViT/AV Transformer Alone Is Not Novel

"Use a Transformer/ViT for steganography" is already broadly explored.

Attention-based and Transformer-based steganography exist.

Recent video steganography also combines INNs with local/non-local attention and adaptive redundancy refinement.

Therefore avoid claiming:

> "Our novelty is using a ViT and attention."

Instead focus on:

> **cross-modal adaptive embedding and cross-modal auxiliary/redundancy prediction for synchronized A/V INN steganography.**

That is the more interesting problem.

---

# 24. Future Direction A Architecture

Potential architecture:

```text
             STEGO VIDEO
                  |
            Video Encoder
                  |
        temporal/spatial tokens
                  |
                  |
                  +---------\
                              \
                               AV Transformer
                              /
                  +---------/
                  |
          STEGO AUDIO
              |
         Audio Encoder
              |
         audio tokens

AV Transformer output
          |
          v
shared multimodal representation H
          |
       +--+--+
       |     |
       v     v
    Audio   Video
    head     head
      |        |
    rA_hat   rV_hat
      |        |
   Audio      Video
   INN^-1    INN^-1
```

Use time-aware embeddings so:

```text
Audio chunk t
Video chunk t
```

are explicitly connected.

However, positional encoding alone does not make the system robust to trimming/dropped chunks. Explicit synchronization information may still be required.

---

# 25. Do Not Force Audio and Video into One Tiny Seed

A tempting idea is:

```text
audio seed + video seed
        |
        v
one combined seed
```

This is probably not the best formulation.

Audio and video redundancy tensors may have very different sizes and semantics.

A better design is:

```text
audio + video
      |
shared AV representation H
      |
  +---+---+
  |       |
rA_hat  rV_hat
```

There is one shared multimodal conditioning state, while each modality has its own output head.

This is more flexible and avoids creating an unnecessary information bottleneck.

---

# 26. Strongest Future Direction-A Experiment

Do not only show that AV attention improves average metrics.

Test cross-modal rescue.

### Experiment 1

```text
audio heavily attacked
video clean
```

Compare:

```text
audio-only redundancy prediction
vs
audio prediction conditioned on video
```

If clean video improves audio redundancy recovery, that demonstrates an actual cross-modal benefit.

### Experiment 2

```text
video heavily attacked
audio clean
```

Compare:

```text
video-only redundancy prediction
vs
video prediction conditioned on audio
```

This creates a meaningful "multimodal recovery" contribution.

---

# 27. Recommended Full Research Roadmap

```text
PHASE 0
Reproduce/freeze existing Audio-INN baseline
        |
        v
PHASE 1
Oracle test:
true r vs random z
clean vs attacked stego
        |
        v
PHASE 2
Deterministic Audio-RPM
        |
        v
PHASE 3
Attack-aware Audio-RPM
        |
        v
PHASE 4
Conditional latent diffusion for r recovery
        |
        v
PHASE 5
RPM -> diffusion refinement
        |
        v
PHASE 6
Inverse-INN secret reconstruction supervision
        |
        v
PHASE 7
Optional stego correction branch
        |
        v
ROBUST AUDIO RESULT
        |
        | if time/results justify
        v
PHASE 8
Integrate with existing Video-INN
        |
        v
PHASE 9
AV Transformer / cross-modal conditioning
        |
        v
PHASE 10
Cross-modal redundancy prediction
        |
        v
MULTIMODAL RESULT
```

---

# 28. Very Important Engineering Strategy

Do **not** build the whole system end-to-end immediately.

Recommended order:

```text
1. Freeze INN.
2. Prove redundancy is predictable.
3. Add deterministic predictor.
4. Prove attack-aware prediction helps.
5. Add diffusion.
6. Prove diffusion improves over deterministic prediction.
7. Add stego repair only if oracle test says it is needed.
8. Fine-tune the complete system only after component-level success.
```

Every stage needs:
- its own checkpoint;
- its own validation set;
- the same evaluation protocol;
- ablation against the previous stage.

This makes the research defensible and debugging tractable.

---

# 29. Current Core Research Claim to Aim For

Do NOT frame it as:

> "We combine StyleGAN, diffusion and INNs."

Use something like:

> We investigate whether the missing auxiliary state of an invertible audio steganography decoder can be inferred from an attacked stego signal using a conditional diffusion model. The model is conditioned on stego-derived features throughout the diffusion process and is trained with both latent redundancy supervision and downstream secret reconstruction through the inverse INN. The goal is robust, graceful secret-audio recovery under channel distortions without transmitting the auxiliary latent.

Potential shorter title/theme:

> **Attack-Aware Diffusion Recovery of Auxiliary Latents for Robust Invertible Audio Steganography**

---

# 30. Core Intuition

Remember this mental model:

```text
             SECRET AUDIO
                  |
                 INN
                  |
          +-------+-------+
          |               |
       stego y            r
          |               |
        attack       normally discarded
          |
          v
         y'
          |
          v
    condition encoder
          |
          v
    diffusion predicts
    useful r_hat
          |
          v
    inverse INN(y', r_hat)
          |
          v
    recovered secret
```

The fundamental separation is:

```text
INN
= organizes/transforms information and provides reversible decoding

RPM
= predicts missing auxiliary information

Diffusion
= handles uncertainty and progressively reconstructs a compatible latent

Optional repair network
= compensates for corruption of the stego input itself
```

Do not make diffusion directly generate the secret at first.

The initial goal is much more controlled:

> **Use diffusion to recover the hidden state needed by the existing invertible decoder.**

---

# 31. Final Priority Decision

### FIRST: Direction B

Work on:

```text
WavInWav
→ predicted redundancy
→ attack-aware redundancy
→ conditional diffusion
→ optional stego repair
```

This has the clearest incremental path to quick results.

### SECOND: Direction A

Only after Direction B works:

```text
existing Video-INN
+
Audio-INN
+
shared AV Transformer
+
cross-modal redundancy prediction
```

The future multimodal contribution should focus on **cross-modal information sharing/recovery**, not merely adding a ViT or positional encoding.

---

# 32. Questions That Must Be Answered Experimentally

Before making strong architectural claims, determine:

1. How much does true `r` improve audio recovery over Gaussian `z`?
2. How much does an RPM improve over Gaussian `z`?
3. Does `r` remain predictable after real codec attacks?
4. Does perfect `r` recovery compensate for attacked stego, or is stego repair also necessary?
5. Does diffusion outperform a deterministic RPM?
6. Does predict-then-diffuse outperform diffusion-from-Gaussian?
7. Does inverse-INN reconstruction supervision improve the diffusion output?
8. Does generated refinement improve perceptual quality without harming content/word accuracy?
9. At what attack severity does critical secret information become irrecoverable?
10. Once audio is robust, does visual context help recover audio redundancy and vice versa?

These experiments should determine the final architecture rather than assuming the largest architecture is necessarily the best one.

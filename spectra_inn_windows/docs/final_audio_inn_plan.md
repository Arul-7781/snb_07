# Final Audio-INN Research Plan

Status: proposed architecture, pending the five decisions in Section 2  
Working name: **SPECTRA-INN** — Survival- and Perceptual-aware Codec-layered Transform Reversible Audio INN

## 1. Executive decision

The rough proposal has the right foundation—time-domain audio, an invertible coupling core, time-frequency losses, psychoacoustic control, attack training, and explicit synchronization—but it should not be implemented as written.

The final design should be a **progressive two-path audio steganography module**:

1. A **clean reversible path** hides a high-fidelity secret-audio enhancement representation with an invertible network. It is optimized for WAV/FLAC or an otherwise unmodified carrier.
2. A **robust base path** hides a lower-rate, perceptually important secret representation plus a short synchronization code. It uses error correction and an attack-trained extractor so that decoding degrades gracefully after compression or signal processing.
3. A **survival-aware psychoacoustic router** allocates the two paths across subbands by balancing inaudibility against expected survival through the target channel.

This is a more credible research contribution than simply combining an INN, DWT, psychoacoustic loss, ECC, and attack augmentation. Those ingredients already occur separately or in combination in recent work.

The main hypothesis is:

> A progressive secret representation, routed using both auditory masking headroom and learned channel survival, can improve the imperceptibility–recovery–robustness Pareto frontier over a uniform or high-frequency-only audio INN.

The design must not claim exact end-to-end reversibility from stego audio alone unless all residual output of the bijection is transmitted. The proposed module is structurally invertible in its clean core; blind secret recovery is learned and approximate.

## 2. Decisions that must be frozen before implementation

The current plan assumes the defaults in the right column.

| Question | Default used by this plan |
|---|---|
| Meaning of reversible | Structurally invertible core and high-fidelity secret recovery; exact original-cover recovery is not required |
| Payload | One time-aligned secret speech/audio stream, not arbitrary files |
| Channel | Lossless delivery is the primary high-capacity mode; MP3/AAC/Opus robustness is a graceful-degradation mode |
| Audio/video relation | Audio and video are components of one secret audiovisual stream and share presentation timestamps |
| Compute | One 20–24 GB RTX 4500-class GPU |

If exact original-cover recovery, exact secret-bit recovery, or full-rate secret recovery after aggressive codecs is mandatory, the scope and achievable payload rate must be changed.

## 3. Evaluation of the rough proposal

### Keep

- A waveform-domain conceal/reveal path.
- Multi-resolution STFT supervision; STFT is a loss/analysis representation, not the sole secret representation.
- Bounded affine coupling and deterministic inverse operations.
- A staged curriculum rather than training all objectives at once.
- Cover, secret, security, robustness, and synchronization metrics reported separately.
- A shared audio/video time map at the system boundary.
- Attack simulation and a held-out real-codec evaluation pipeline.

### Change

1. **Separate clean inversion from attacked decoding.** A codec breaks the exact inverse relation and affine inverse blocks can amplify small perturbations. Use the reverse INN for the clean path and a small non-invertible, attack-trained extractor for the robust path.
2. **Do not hide everything in high frequencies.** High bands can be perceptually convenient but are the first content discarded by codecs. The router should use mid/high bands according to both masking headroom and measured survival.
3. **Use an accessible conditioning signal.** Any condition used by an exact inverse must be reproducible at the decoder. Compute routing from a preserved/low-distortion anchor subband or from the current reversible state—not from unavailable original-cover features.
4. **Do not re-embed the whole INN residual with ECC.** That creates a recursive capacity problem. Regularize the clean-path residual to a keyed prior; apply ECC only to the explicitly digital robust base payload and synchronization header.
5. **Do not overlap-average independently encoded chunks.** Averaging breaks exact cycle behavior. Use fixed non-overlapping core segments with receptive-field context that is cropped, or an explicitly invertible lapped transform.
6. **Use a real cryptographic boundary.** Sample scrambling is obfuscation, not encryption. If the secret becomes a bitstream, encrypt it with an authenticated cipher before ECC and embedding. Keep key management outside the neural network.
7. **Let the container timestamps perform normal A/V sync.** The embedded synchronization signal is a fallback for trimming, concatenation, remuxing, or metadata loss—not the primary clock.

### Remove or avoid claiming

- “Exact reconstruction from stego audio” when the forward residual is discarded.
- “STFT is inherently lossy.” A properly configured STFT/iSTFT can be perfect reconstruction; magnitude-only representations and inconsistent predicted spectra are the actual problem.
- “Random latent, predicted latent, and ECC-protected latent” as three simultaneous reverse strategies. Select one mechanism per operating mode.
- A fixed `1,2,4,8,16,8,4,2,1` dilation schedule before receptive-field and memory profiling.
- Adversarial training as part of the first prototype.
- Int8 waveform audio as a normal media format; evaluate PCM bit-depth reduction explicitly instead.

## 4. Why the original reversibility claim needs correction

Let the forward INN map

```text
(cover, secret) -> (stego, residual)
```

The two sides have equal dimensionality, so this internal mapping can be bijective. If only `stego` is transmitted and `residual` is discarded, the receiver no longer has the complete output of the bijection. Replacing `residual` with Gaussian noise or a prediction can yield a good recovered secret, but it is not exact inversion of the original pair.

The implementation and paper should report three different properties:

- **Numerical cycle error:** inverse(forward(cover, secret)) when the true residual is retained.
- **Blind clean recovery:** inverse(stego, keyed prior) when only the stego signal is available.
- **Attacked recovery:** robust extractor(attack(stego)).

This distinction prevents an easily challenged “lossless/reversible” claim.

## 5. Proposed system boundary

### Inputs

- Cover audio `c`, with original sample rate/channel metadata and presentation timestamp.
- Secret audio `s`, aligned to the same time interval.
- Segment record: `stream_id`, `segment_index`, `start_pts`, `duration`, `mode`.
- Secret key handle; the raw key should never be stored in a model checkpoint.

### Outputs

- Stego audio `y` with exactly the same sample count, channel count, nominal sample rate, and timestamps as the cover.
- Encoder diagnostics: payload allocation, true-peak margin, masking violations, latent statistics.
- Decoder outputs: recovered secret, confidence mask, recovered segment counter, and timing confidence.

### Initial operating points

Use two configurations rather than forcing one configuration to serve incompatible goals.

| Configuration | Cover | Secret | Purpose |
|---|---|---|---|
| Baseline/comparison | 16 kHz mono, 2 s | 16 kHz mono, 2 s | Reproduce and compare with WavInWav-style work |
| Integration target | 48 kHz stereo, 1 s | 16 kHz mono, 1 s | Real audiovisual carrier with a lower-dimensional speech secret |

The 48 kHz carrier must not simply be downsampled to 16 kHz, because that would audibly and statistically alter the carrier and make later video integration unrealistic.

## 6. SPECTRA-INN architecture

### 6.1 Secret representation

Create a progressive representation of each secret segment:

```text
secret audio
  -> frozen low-bitrate audio codec or learned analysis transform
  -> base representation B (timing, coarse spectral envelope, intelligibility)
  -> enhancement representation E (speaker detail, phase, high-frequency residual)
```

Recommended first implementation:

- Start with a lightweight frozen neural audio codec and precompute its representations.
- Use the lowest-rate codebook(s) or a learned bottleneck as `B`.
- Define `E` as the remaining codec codebooks or a waveform residual.
- Do not train the codec jointly on the single GPU in the first study.

This makes graceful degradation explicit: an attacked carrier may recover only `B`, while a clean carrier recovers `B + E`.

Run a capacity preflight before committing to a codec. Measure the actual base-layer bits per second including framing, authenticated encryption, and ECC. If the base payload cannot be hidden at the required cover quality, reduce its rate or move part of it to the video channel; do not silently increase audio distortion.

Use the following rate ledger as a reality check:

| Payload class | Approximate order of magnitude | Interpretation |
|---|---:|---|
| Pilot/sync watermark | 5–16 bit/s | Realistic first robustness target |
| Transcript/semantic base | 50–300 bit/s | Research target; reconstruction is partly generative |
| Very-low-rate neural audio codec | 1.5 kbit/s or more before ECC | Ambitious robust target; likely needs video spillover |
| 16 kHz, 16-bit mono PCM | 256 kbit/s | Clean high-capacity analog hiding only; not a credible codec-robust target |

Recent robust watermark benchmarks commonly operate near 5 bit/s, so a protected codec base layer is not “small” relative to demonstrated robust watermark capacity. The project should first prove a sync channel, then a semantic/base channel at increasing rates. Full neural-codec payload is a stretch goal, not an assumed deliverable.

Amortize authentication tags, headers, and parity across a 4–8 second super-segment even if the logical A/V time grid remains one second. For the enhancement path, convert discrete residual codebooks to continuous embeddings (or use a waveform residual) and explicitly measure token/nearest-neighbor errors after clean inversion.

### 6.2 Reversible cover analysis

Apply a perfect-reconstruction 1D lifting wavelet or learned lifting transform to the cover. Use three scales initially.

The resulting channels are divided into:

- `A`: an anchor/conditioning band kept unchanged or very lightly modified.
- `C_r`: codec-survivable carrier bands for the robust base payload.
- `C_e`: masking-rich carrier bands for clean enhancement hiding.

Haar DWT is the reproducible baseline. A learned lifting transform is an ablation, not an MVP dependency.

### 6.3 Dual-objective router

For every time–subband tile, estimate:

- `M(c)`: perceptual masking headroom—the maximum safe perturbation energy.
- `Q(c, channel)`: probability or expected quality of payload survival after the target attack family.

Compute an allocation map under a fixed distortion/payload budget:

```text
R = normalize((M + eps)^alpha * (Q + eps)^beta)
```

Use two router temperatures or budgets:

- `R_base`: favors `Q`, routing the protected base representation to stable mid-bands.
- `R_enh`: favors `M`, routing enhancement information to highly masked regions.

`Q` can begin as an empirical lookup table measured from the attack pipeline. A learned survival predictor is introduced only after the deterministic router baseline works.

To preserve clean reversibility, router inputs must be derivable in both directions. The safest MVP is to derive `R` from anchor features `A` that the encoder does not modify.

### 6.4 Clean invertible enhancement core

Pack `C_e` and the continuous enhancement representation `E` to compatible shapes, then apply 8–12 reversible blocks. A block uses a standard bounded affine coupling:

```text
u' = u + phi(v, A, R_enh)
log_scale = clamp * tanh(rho(u', A, R_enh))
v' = v * exp(log_scale) + eta(u', A, R_enh)
```

The inverse is analytic:

```text
v = (v' - eta(u', A, R_enh)) * exp(-log_scale)
u = u' - phi(v, A, R_enh)
```

Subnetworks should use compact depthwise-separable 1D residual blocks, gated activations, and a measured receptive field. Use invertible channel permutations or LU-parameterized `1x1` mixing between blocks.

The forward output is `(C_e_stego, r)`. Regularize `r` toward a keyed, simple prior and discard it for blind decoding. During unit tests retain `r` to verify the exact cycle. During real clean decoding use a deterministic key-derived prior sample `z_k`, not an unkeyed random sample and not a second embedded copy of `r`.

### 6.5 Robust base embedder and extractor

The robust path is intentionally not forced through the analytic inverse after an attack.

1. Serialize and, if required, authenticate/encrypt `B`.
2. Add CRC and configurable BCH/LDPC-style error protection.
3. Interleave across time tiles within a segment or short super-segment.
4. Embed into `C_r` with a bounded residual controlled by `R_base`.
5. Decode using a small attack-trained extractor with soft-bit outputs and confidence.

The robust extractor should also predict a synchronization pilot/counter. Keep this head small enough that it can scan long audio efficiently.

For an initial scientific result, compare:

- robust base embedded before the INN;
- robust base embedded after the INN;
- joint residual composition with a shared distortion budget.

The ordering is an empirical question because the clean INN may either preserve or interfere with the robust signal.

### 6.6 Clean and attacked decode modes

**Clean mode**

```text
stego -> wavelet analysis -> recover B -> keyed z_k + inverse INN -> recover E
      -> codec/synthesis decoder(B, E) -> high-fidelity secret
```

**Robust mode**

```text
attacked stego -> synchronization/alignment -> robust extractor -> ECC -> B_hat
               -> codec/synthesis decoder(B_hat, missing E) -> coarse/intelligible secret
```

An optional refinement network may estimate missing enhancement data from `B_hat`, but its output must be labeled generative reconstruction rather than recovered hidden information.

## 7. Audio/video synchronization contract

Synchronization belongs primarily in the multimodal orchestrator, not inside the INN.

1. Use the media container's integer presentation timestamps as the source of truth.
2. Divide the carrier into one-second logical segments identified by `(stream_id, segment_index, start_pts)`.
3. Preserve audio sample count exactly during hiding; never time-stretch merely to fit a payload.
4. Align secret-video frame groups and secret-audio sample ranges to the same logical segment records.
5. Add a keyed pilot and a short rolling segment counter to both modalities as a fallback alignment signal.
6. After trimming or drift, fit a clock relation `t_secret = a * t_carrier + b` from recovered anchors. Correct drift only after payload decoding.
7. Put cross-modal erasure coding in the orchestrator. The video channel can carry parity for critical audio base packets when video has spare capacity, and vice versa only at a deliberately low rate.

Do not place large JSON metadata or a full latent tensor into the carrier. Normal, non-secret timestamps may remain in the media container.

## 8. Losses

Use normalized loss groups and log every component. Initial weights are hyperparameters, not architectural constants.

### Cover imperceptibility

- Sample-domain Charbonnier loss.
- Multi-resolution complex STFT plus magnitude loss.
- A-weighted or ERB-band perturbation energy.
- Masking-threshold violation loss.
- LUFS deviation and true-peak/clipping penalty.
- Extra penalty in silence/near-silence.

### Clean secret recovery

- Base token cross-entropy or bottleneck reconstruction loss.
- Enhancement latent loss.
- Waveform L1/Charbonnier after synthesis.
- Multi-resolution STFT loss.
- SI-SDR/STOI-oriented term for speech only.

### Invertible-core regularization

- True-residual forward/reverse cycle error for diagnostics.
- Prior matching for discarded residual `r`.
- Scale/Jacobian stability penalty.
- Key-prior consistency loss for blind clean decoding.

### Robustness and synchronization

- Soft-bit/base-token loss after sampled attacks.
- ECC-aware packet success objective or differentiable surrogate.
- Pilot detection and segment-counter loss under random offsets/trims.
- Consistency between clean base decode and attacked base decode.

### Security

Introduce a separately trained steganalyzer late. Alternate rather than jointly destabilizing the entire system from the start. Security evaluation must include steganalyzers trained on held-out cover sources and attacks not used to train the embedder.

## 9. Attack curriculum

Use attack categories rather than sampling every transform uniformly. Keep differentiable approximations for training and real tools/codecs for validation.

1. Identity, gain, polarity inversion, PCM quantization.
2. Gaussian/background noise, filtering, EQ, dynamic-range compression, limiting.
3. Small time shift, crop, local dropout, resampling, mild time stretch.
4. MP3, AAC, Opus/OGG at realistic bitrates.
5. Neural codecs only as an out-of-distribution stress test unless the deployment channel actually uses them.

Begin robust training with the clean INN frozen. Train the robust base extractor, then jointly fine-tune only the router and the last few embedding blocks at a lower learning rate. This avoids destroying the clean solution.

## 10. Training plan for one RTX 4500

Target a 6–12 million parameter audio module before any steganalyzer. Use automatic mixed precision, gradient accumulation, activation checkpointing, and precomputed codec representations.

| Stage | Goal | Suggested scope | Exit criterion |
|---|---|---|---|
| 0 | Data and channel harness | segmentation, timestamps, metrics, real-codec round trips | deterministic tests pass |
| 1 | Reproduce a baseline | 16 kHz mono, 8 coupling blocks, no router | baseline quality is credible |
| 2 | Clean SPECTRA-INN | anchor bands, deterministic router, clean path | stable blind clean recovery and no clipping |
| 3 | Progressive secret | base/enhancement representation | clean `B+E` beats base-only clearly |
| 4 | Robust base path | ECC, pilot, attack-trained extractor | packet recovery curve measured |
| 5 | Learned survival router | estimate `Q` and jointly tune | Pareto improvement over fixed routing |
| 6 | Security and A/V integration | steganalysis, PTS contract, cross-modal parity | end-to-end synchronized demo |

Practical settings:

- Baseline crops: 2 s at 16 kHz, batch 2–4.
- Integration crops: 0.5–1 s at 48 kHz stereo, micro-batch 1–2, accumulated effective batch 8–16.
- Coupling blocks: start at 8, width 48 or 64; scale only after profiling.
- Keep a fixed validation suite of short clips that runs in minutes.
- Run the full real-codec and steganalysis suite only at milestones.

Do not promise a wall-clock schedule until one thousand representative training steps are profiled on the actual GPU. Use that measurement to estimate each stage.

## 11. Data plan

Train and test across carrier domains because a model trained only on clean speech can exploit domain-specific shortcuts.

- Secret: speech first; VCTK and LibriSpeech enable comparison with recent audio-in-audio work.
- Covers: balanced speech, music, and environmental audio.
- Split by speaker, recording, song/track, and source dataset—not random chunks—to avoid leakage.
- Reserve at least one complete cover dataset and one attack family for out-of-distribution testing.
- Track licenses and resampling history for every corpus.

Add music-secret and general-audio-secret experiments only after the speech-secret system is stable.

## 12. Evaluation and required baselines

### Cover quality

- SNR and SI-SDR.
- Multi-resolution STFT distance and log-spectral distance.
- ViSQOL or another domain-appropriate perceptual metric.
- LUFS change, true peak, clipping rate, silence distortion.
- MUSHRA/ABX listening study on a preregistered subset.

### Secret quality

- Clean and attacked SI-SDR/SNR.
- STOI and PESQ for speech, with their domain/sample-rate limitations stated.
- Speaker-embedding similarity and ASR word error rate for speech.
- Base-packet BER, post-ECC packet success, and enhancement recovery error.

### Robustness

- Quality-versus-attack-strength curves, not a single average.
- Separate results for seen and unseen attacks.
- Segment acquisition latency, synchronization success, PTS error, and recovery after leading/middle trims.
- Clean, MP3/AAC/Opus, resampling, filtering, noise, limiting, polarity, time shift/stretch, and neural-codec stress tests.

### Security

- Cover-versus-stego ROC-AUC and detection error at fixed false-positive rates.
- Cross-dataset steganalysis.
- Known-model and unknown-model attacker settings.
- Wrong-key recovery quality and false pilot detection.

### Baselines

- A WavInWav-style time-domain INN reproduction.
- InvASNet-style Haar/high-frequency routing if implementation access permits.
- A matched non-invertible encoder/decoder with similar parameter count.
- Uniform routing and masking-only routing.
- Conventional LSB only as a clean-capacity sanity check.
- A robust watermark model such as WavMark/IDEAW only for the low-rate sync/base channel, not as a full-audio payload comparison.

## 13. Ablations needed to support novelty

1. No progressive split versus base/enhancement split.
2. Uniform, masking-only, survival-only, and joint `M x Q` routing.
3. High-frequency-only versus codec-survivable mid-band routing.
4. Analytic inverse after attacks versus separate robust extractor.
5. Unprotected base versus ECC/interleaving.
6. Original-cover conditioning versus anchor-derived invertible conditioning.
7. Random prior versus key-derived prior.
8. Audio-only recovery versus cross-modal parity for critical base packets.

The central result should be a Pareto plot with cover quality on one axis and recovered-secret quality/packet success on the other, with separate curves for clean and attacked carriers.

## 14. Milestones and stop/go gates

### M1 — honest baseline

- Reproduce comparable clean quality on 16 kHz speech.
- Verify true-residual cycle error and blind-prior recovery separately.
- **Stop** if the baseline cannot generalize to unseen speakers/datasets.

### M2 — final carrier format

- Move to 48 kHz stereo cover and 16 kHz mono secret.
- Preserve duration, channels, peaks, and timestamps.
- **Stop** if ordinary remux/resample destroys clean recovery before robustness work begins.

### M3 — progressive representation

- Show that base-only recovery is intelligible and `B+E` is materially better.
- Complete a measured payload-capacity budget.
- **Stop or move payload to video** if protected base bitrate is incompatible with imperceptibility.

### M4 — survival-aware router

- Beat masking-only and high-frequency-only routing on at least one preregistered robustness-quality frontier without regressing all other domains.
- This is the main publishable method gate.

### M5 — synchronized audiovisual prototype

- Decode a secret A/V segment with preserved PTS.
- Recover alignment after a defined leading trim and one missing segment.
- Demonstrate cross-modal parity only after both independent modules work.

## 15. Concrete repository implementation order

```text
audio_inn/
  configs/
  data/
    segmenter.py
    pairing.py
  transforms/
    lifting.py
    stft.py
    psychoacoustic.py
  models/
    coupling.py
    clean_inn.py
    router.py
    robust_embedder.py
    robust_extractor.py
    secret_codec.py
  channel/
    differentiable.py
    real_codecs.py
  sync/
    packet.py
    pilot.py
    clock_fit.py
  losses/
  metrics/
  tests/
    test_invertibility.py
    test_length_pts.py
    test_codec_roundtrip.py
    test_sync_reacquisition.py
  train.py
  evaluate.py
```

Implementation sequence:

1. Segment/time contract and evaluation harness.
2. Haar/lifting transform with exact tests.
3. Small WavInWav-style baseline and matched encoder/decoder baseline.
4. Anchor-derived masking router and clean INN.
5. Progressive secret representation with a written capacity report.
6. Robust base embedder/extractor, ECC, and pilot.
7. Empirical then learned survival model.
8. Steganalysis and listening evaluation.
9. Audio/video orchestrator integration.

## 16. Literature position

The novelty claim must be framed against at least these nearby works:

- [InvASNet / Hiding speech in music files](https://doi.org/10.1016/j.jisa.2024.103951): INN audio-in-audio, Haar decomposition, and high-frequency embedding.
- [WavInWav](https://arxiv.org/abs/2510.02915): time-domain full-secret speech hiding, invertible coupling, time-frequency loss, noise layers, and discarded Gaussianized residual.
- [IDEAW](https://aclanthology.org/2024.emnlp-main.258/): invertible dual embedding of message and locating code with attack training.
- [HIA-Net](https://doi.org/10.1016/j.jisa.2026.104382): auditory masking-driven progressive invertible image-in-audio hiding.
- [RAW-Bench](https://www.isca-archive.org/interspeech_2025/ozer25_interspeech.html): broad real-world attack evaluation and evidence that neural codecs remain especially difficult.
- [LF-VSN](https://arxiv.org/abs/2304.12300): scalable multi-secret video hiding and the likely video-side architectural reference.

Therefore, the defensible contribution is not “an INN plus psychoacoustic loss and robustness.” It is the tested combination of:

1. a **progressive base/enhancement secret-audio representation**;
2. an **invertibility-safe router jointly optimizing masking and channel survival**;
3. a **clean analytic inverse plus protected attack-trained base decoder**; and
4. a **timestamped cross-modal recovery contract** that allows the video channel to protect critical audio packets.

## 17. Final recommendation

Build the smallest baseline first, but design the interfaces around the final 48 kHz audiovisual carrier and shared PTS from day one. Treat clean reversibility and channel robustness as two measured operating modes. Make the progressive secret split and dual-objective router the research centerpiece. Keep ECC, encryption, synchronization, and cross-modal parity explicit and rate-accounted rather than describing them as abstract latent “redundancy.”

This plan is feasible to prototype on one RTX 4500-class GPU because the secret codec can remain frozen, the INN can stay below roughly 12 million parameters, and robustness can be trained as a small second path after the clean core converges.

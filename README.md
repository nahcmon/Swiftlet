# Swiftlet

Fast, on-device speech-to-text for Apple silicon Macs, as a Swift package.

Swiftlet runs [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-coreml) (FastConformer-TDT, 25 European languages) on its own small inference engine instead of Core ML:

- **Encoder on the GPU** with hand-written Metal kernels that read 4-bit palette weights straight from a memory-mapped file
- **Decoder on the CPU** with NEON fp16 matrix-vector products spread over the performance cores
- **Core ML only for the log-mel front end**, on the CPU

Short dictations finish in about 25–60 ms, 46 minutes of audio take 23 s, and the model is ready 0.15 s after launch. Weights: [nahcmon/fluid-speech-pro-swiftlet-4bit](https://huggingface.co/nahcmon/fluid-speech-pro-swiftlet-4bit) on Hugging Face.

## Getting started

### 1. What you need

- A Mac with Apple silicon (M1 or newer) on macOS 15 or later
- About 400 MB free disk space for the model
- Xcode 16 or later, or just the command line tools:

```bash
xcode-select --install
```

### 2. Try it from the command line

```bash
git clone https://github.com/nahcmon/Swiftlet
cd Swiftlet
swift build -c release
```

Download the model once (361 MB from [Hugging Face](https://huggingface.co/nahcmon/fluid-speech-pro-swiftlet-4bit), every file checked against its SHA-256). It goes to `~/Library/Application Support/Swiftlet/fluid-speech-pro-4bit`:

```bash
.build/release/swiftlet download
```

Transcribe any audio file AVFoundation can read (WAV, M4A, MP3, CAF, AIFF, ...), at any sample rate:

```bash
.build/release/swiftlet transcribe memo.m4a
```

Several files at once, with load and per-file timings on stderr:

```bash
.build/release/swiftlet transcribe --time memo.m4a call.wav lecture.mp3
```

To keep the model somewhere else, pass the folder to `download` and `--model <folder>` to `transcribe`. The language is detected automatically.

### 3. Use it in your app

In Xcode: **File › Add Package Dependencies…**, enter `https://github.com/nahcmon/Swiftlet`, and add the `Swiftlet` library to your target. Or in `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/nahcmon/Swiftlet", from: "0.1.0"),
],
targets: [
    .target(name: "MyApp", dependencies: ["Swiftlet"]),
]
```

```swift
import Swiftlet

// Downloads the model the first time (361 MB); later calls return at once.
let folder = try await ModelDownloader.download { done, total in print(done * 100 / total, "%") }

// Load once and keep it; loading takes about 0.15 s.
let engine = try SpeechEngine(directory: folder)

let text = try engine.transcribe(contentsOf: URL(fileURLWithPath: "memo.m4a"))   // any file AVFoundation reads
let live = try engine.transcribe(samples: samples)                                // 16 kHz mono Float, -1 ... 1
let raw  = try engine.transcribe(pcm: data)                                      // 16 kHz mono Int16 little-endian
```

Things to know:

- **Build in Release** when you measure speed. Debug builds leave the NEON decoder unoptimized and several times slower.
- `SpeechEngine` is **not thread-safe**: use it from one thread or serial queue at a time, off the main thread.
- A **sandboxed app** needs the outgoing-network entitlement for the download, or ship the model folder with your app and pass its URL to `SpeechEngine(directory:)`.
- `ModelDownloader.isDownloaded()` tells you whether the model is already there.

### Live dictation

Long recordings are cut into windows of at most 15 s at the quietest moment near each window's end. While someone is still talking, call `prefetch(pcm:)` with the audio so far (for example every few seconds). Windows whose cut can no longer move are decoded ahead, and the final `transcribe(pcm:)` reuses them only if their audio is byte-identical, so the transcript is never different, just ready sooner. Call `discardPrefetched()` if the recording is thrown away.

## Benchmarks

Measured on an M1 Pro (16 GB) with 234 clips, 46 minutes of English and German audio (best of two runs per clip), against the original fp16 Core ML release of Fluid Speech Pro running on the Neural Engine.

### Speed

| Audio length | Clips | Original (Core ML, fp16) | Swiftlet 4-bit | Speedup |
|---|---|---|---|---|
| 0–3 s | 19 | 58 ms | 26 ms | 2.2× |
| 3–6 s | 60 | 70 ms | 42 ms | 1.7× |
| 6–10 s | 59 | 85 ms | 61 ms | 1.4× |
| 10–15 s | 51 | 108 ms | 99 ms | 1.1× |
| 15–30 s | 40 | 183 ms | 145 ms | 1.3× |
| 2-minute file | 1 | 1.18 s | 0.95 s | 1.2× |
| 3-minute file | 1 | 1.58 s | 1.41 s | 1.1× |

| | Original | Swiftlet 4-bit |
|---|---|---|
| All 46 min of audio | 29.2 s | 22.9 s |
| Real-time factor | 0.0105 (95× real time) | 0.0082 (122× real time) |
| Model load, first launch | 20.6 s (Neural Engine compile) | 0.15 s |
| Model load, later | 0.23 s | 0.13 s |

Times are medians per clip, from audio in memory to text.

### Accuracy (word error rate, lower is better)

| Audio | Original | Swiftlet 4-bit |
|---|---|---|
| **All clips** | **6.72%** | **6.75%** |
| English, clean | 2.66% | 3.11% |
| English, other speakers and accents | 4.95% | 4.69% |
| English, quiet | 2.84% | 2.58% |
| English, noisy | 5.34% | 5.77% |
| German, read | 6.32% | 7.94% |
| German, phone quality | 20.23% | 19.38% |
| Jargon | 5.03% | 5.03% |
| Long files (2–3 min) | 6.18% | 5.13% |

175 of 234 transcripts are word-for-word identical to the original. The differences come from the 4-bit encoder weights; German read speech loses the most.

### Memory and disk

| | Original | Swiftlet 4-bit |
|---|---|---|
| Process memory, peak (5 files incl. 2–3 min) | 104 MB RSS / 84 MB footprint | 92 MB RSS / 92 MB footprint |
| Weights | held by the Neural Engine service | 307 MB file-backed mapping (macOS can page it out and back in) |
| Size on disk | 622 MB | 344 MB |

## How it works

| Stage | Where | Details |
|---|---|---|
| Log-mel features | CPU (Core ML) | Upstream preprocessor, unchanged |
| Encoder (24 FastConformer layers) | GPU (Metal) | `SpeechKernels.swift`: 4-bit palette GEMMs with simdgroup matrix tiles and split-K, fused bias, activation and residual epilogues, depthwise convolutions, relative-position attention. Weights stay 4-bit and file-backed; only the actual audio frames are computed, never padding |
| TDT decoding | CPU (NEON) | `SpeechDecoder.swift` + `CSwiftlet`: greedy TDT search, fp16 weights with fp32 accumulation (FMLAL), output head split across performance cores on spinning worker threads |

## Re-exporting the weights

```bash
pip install coremltools==9.0 numpy
python3 Tools/palettize_encoder.py <upstream Encoder.mlmodelc> pal4/Encoder.mlmodelc   # ~10 minutes
python3 Tools/export_model.py pal4/Encoder.mlmodelc <upstream model folder> out/
```

## License

Code: MIT (see `LICENSE`). Model weights: CC BY 4.0, see the [model card](https://huggingface.co/nahcmon/fluid-speech-pro-swiftlet-4bit).

Fluid Speech Pro by FluidVoice. Built on NVIDIA Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra. Swiftlet is not affiliated with or endorsed by FluidVoice, NVIDIA or Moondream.

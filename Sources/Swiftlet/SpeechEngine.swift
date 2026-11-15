import CoreML
import CryptoKit
import Foundation

/// Fluid Speech Pro (FastConformer-TDT, 25 European languages) on Swiftlet, a Metal + NEON inference engine for Apple silicon.
///
/// The Core ML preprocessor computes the log-mel features on the CPU, SpeechEncoder runs the encoder on the GPU
/// with 4-bit weights, and SpeechDecoder runs greedy TDT decoding on the CPU.
///
/// Audio is decoded in windows of at most 15 s. Long recordings are cut at the quietest 80 ms near the end of
/// each window, because the model can drop a whole window that starts mid-word. Each window is decoded with
/// greedy TDT search and the tokens are concatenated.
///
/// Not thread-safe: use one engine from one thread or serial queue at a time.
public final class SpeechEngine {
    public struct Failure: LocalizedError {
        public let message: String
        public var errorDescription: String? { message }
    }

    /// Input sample rate in Hz.
    public static let sampleRate = 16_000
    static let windowSamples = 240_000
    static let frameSamples = 1_280
    static let preprocessorFile = "Preprocessor.mlmodelc"
    static let encoderFile = "encoder.bin"
    static let decoderFile = "decoder.bin"
    static let vocabularyFile = "parakeet_v3_vocab.json"
    private static let melBins = 128
    /// Cuts are searched in the last ~5 s of a window.
    private static let searchFrames = 62

    /// The model folder this engine was loaded from.
    public let directory: URL
    private let preprocessor: MLModel
    private let encoder: SpeechEncoder
    private let decoder: SpeechDecoder
    private let vocabulary: [String]
    private let audio: MLMultiArray
    private let audioLength: MLMultiArray
    /// Windows decoded ahead of time, keyed by start sample; reused only for byte-identical audio.
    private var prefetched: [Int: (end: Int, digest: SHA256Digest, tokens: [Int])] = [:]

    /// Throws if `directory` is missing a model file.
    public static func validate(_ directory: URL) throws {
        let manager = FileManager.default
        let files = ["coremldata.bin", "model.mil", "weights/weight.bin"].map { "\(preprocessorFile)/\($0)" } + [encoderFile, decoderFile]
        for file in files where !manager.fileExists(atPath: directory.appendingPathComponent(file).path) {
            throw Failure(message: "Speech model file missing: \(file)")
        }
        let vocabulary = directory.appendingPathComponent(vocabularyFile)
        let size = (try? manager.attributesOfItem(atPath: vocabulary.path)[.size] as? NSNumber)?.intValue ?? 0
        guard size > 100_000 else { throw Failure(message: "Speech model vocabulary is missing or incomplete.") }
    }

    /// Loads the model in `directory` (see ModelDownloader). Takes about 0.15 s; keep the engine around.
    public init(directory: URL) throws {
        #if !arch(arm64)
        throw Failure(message: "Swiftlet requires an Apple silicon Mac.")
        #endif
        guard ProcessInfo.processInfo.isOperatingSystemAtLeast(OperatingSystemVersion(majorVersion: 15, minorVersion: 0, patchVersion: 0)) else {
            throw Failure(message: "Swiftlet requires macOS 15 or later.")
        }
        try Self.validate(directory)
        self.directory = directory
        let cpu = MLModelConfiguration(); cpu.computeUnits = .cpuOnly
        preprocessor = try MLModel(contentsOf: directory.appendingPathComponent(Self.preprocessorFile), configuration: cpu)
        encoder = try SpeechEncoder(weightsURL: directory.appendingPathComponent(Self.encoderFile))
        decoder = try SpeechDecoder(url: directory.appendingPathComponent(Self.decoderFile))

        let data = try Data(contentsOf: directory.appendingPathComponent(Self.vocabularyFile))
        guard let entries = try JSONSerialization.jsonObject(with: data) as? [String: String] else {
            throw Failure(message: "Speech model vocabulary is unreadable.")
        }
        var vocabulary = [String](repeating: "", count: SpeechDecoder.blank + 1)
        for (key, piece) in entries { if let id = Int(key), vocabulary.indices.contains(id) { vocabulary[id] = piece } }
        self.vocabulary = vocabulary

        audio = try MLMultiArray(shape: [1, NSNumber(value: Self.windowSamples)], dataType: .float32)
        audioLength = try MLMultiArray(shape: [1], dataType: .int32)
    }

    /// `pcm` is 16 kHz mono Int16 little-endian. `shouldStop` is checked between windows.
    public func transcribe(pcm: Data, shouldStop: () -> Bool = { false }) throws -> String {
        defer { prefetched.removeAll() }
        let samples = Self.floats(pcm)
        var tokens: [Int] = []
        for window in Self.windows(count: samples.count, samples: samples) {
            if shouldStop() { throw CancellationError() }
            let digest = SHA256.hash(data: pcm[pcm.startIndex + window.lowerBound * 2 ..< pcm.startIndex + window.upperBound * 2])
            if let cached = prefetched[window.lowerBound], cached.end == window.upperBound, cached.digest == digest {
                tokens += cached.tokens
            } else {
                tokens += try autoreleasepool { try decode(samples[window]) }
            }
        }
        return text(tokens)
    }

    /// Decodes the windows of a growing recording whose cut points can no longer change, so the final
    /// `transcribe` only has the tail left. Never changes the transcript.
    public func prefetch(pcm: Data, shouldStop: () -> Bool = { false }) throws {
        let samples = Self.floats(pcm)
        for window in Self.windows(count: samples.count, samples: samples).dropLast() {
            if shouldStop() { return }
            let digest = SHA256.hash(data: pcm[pcm.startIndex + window.lowerBound * 2 ..< pcm.startIndex + window.upperBound * 2])
            if let cached = prefetched[window.lowerBound], cached.end == window.upperBound, cached.digest == digest { continue }
            let tokens = try autoreleasepool { try decode(samples[window]) }
            prefetched[window.lowerBound] = (window.upperBound, digest, tokens)
        }
    }

    public func discardPrefetched() { prefetched.removeAll() }

    /// `samples` are 16 kHz mono floats in -1 ... 1.
    public func transcribe(samples: [Float], shouldStop: () -> Bool = { false }) throws -> String {
        try transcribe(pcm: Self.pcm(samples), shouldStop: shouldStop)
    }

    /// Any audio file AVFoundation can read (WAV, M4A, MP3, CAF, ...), at any sample rate and channel count.
    public func transcribe(contentsOf url: URL) throws -> String {
        try transcribe(pcm: AudioFile.pcm16k(url))
    }

    static func pcm(_ samples: [Float]) -> Data {
        var data = Data(count: samples.count * 2)
        data.withUnsafeMutableBytes { raw in
            for (index, sample) in samples.enumerated() {
                let value = Int16(max(-32_768, min(32_767, (sample * 32_768).rounded())))
                raw.storeBytes(of: value.littleEndian, toByteOffset: index * 2, as: Int16.self)
            }
        }
        return data
    }

    // MARK: - Windows

    /// Window boundaries. Every window but the last depends only on audio inside it, so a longer recording
    /// with the same beginning yields the same leading windows.
    static func windows(count: Int, samples: [Float]) -> [Range<Int>] {
        guard count > windowSamples else { return [0..<count] }
        let frameCount = count / frameSamples
        var energy = [Float](repeating: 0, count: frameCount)
        samples.withUnsafeBufferPointer { buffer in
            for frame in 0..<frameCount {
                var sum: Float = 0
                for index in (frame * frameSamples)..<((frame + 1) * frameSamples) { sum += buffer[index] * buffer[index] }
                energy[frame] = sum
            }
        }
        // Smoothed over three frames so one quiet frame inside a word doesn't win.
        func smoothed(_ frame: Int) -> Float {
            energy[max(0, frame - 1)...min(frameCount - 1, frame + 1)].reduce(0, +)
        }
        let maxFrames = windowSamples / frameSamples - 1
        var windows: [Range<Int>] = []
        var start = 0
        while count - start > windowSamples {
            let first = start / frameSamples
            let high = first + maxFrames, low = high - searchFrames
            var best = high, quietest = Float.infinity
            for frame in low..<min(high, frameCount) {
                let value = smoothed(frame)
                if value < quietest { quietest = value; best = frame }
            }
            let cut = best * frameSamples + frameSamples / 2
            windows.append(start..<cut)
            start = cut
        }
        windows.append(start..<count)
        return windows
    }

    // MARK: - Decoding

    private func decode(_ samples: ArraySlice<Float>) throws -> [Int] {
        guard !samples.isEmpty else { return [] }
        let input = audio.dataPointer.bindMemory(to: Float.self, capacity: Self.windowSamples)
        input.initialize(repeating: 0, count: Self.windowSamples)
        samples.withUnsafeBufferPointer { input.update(from: $0.baseAddress!, count: $0.count) }
        audioLength[0] = NSNumber(value: samples.count)
        let features = try preprocessor.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": audio, "audio_length": audioLength]))
        let mel = try value(features, "mel")
        let melFrames = min(try value(features, "mel_length")[0].intValue, SpeechEncoder.maxMel, mel.shape[2].intValue)
        guard melFrames >= 1, mel.shape[1].intValue == Self.melBins else { return [] }

        // Only the frames of the audio go to the GPU; the preprocessor pads to 15 s.
        let bins = mel.strides[1].intValue, steps = mel.strides[2].intValue
        let target = encoder.mel.contents().bindMemory(to: Float.self, capacity: Self.melBins * SpeechEncoder.maxMel)
        switch mel.dataType {
        case .float32:
            let source = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
            for bin in 0..<Self.melBins {
                for step in 0..<melFrames { target[bin * SpeechEncoder.maxMel + step] = source[bin * bins + step * steps] }
            }
        #if arch(arm64)  // Float16 doesn't exist on Intel, where init already refuses to run
        case .float16:
            let source = mel.dataPointer.bindMemory(to: Float16.self, capacity: mel.count)
            for bin in 0..<Self.melBins {
                for step in 0..<melFrames { target[bin * SpeechEncoder.maxMel + step] = Float(source[bin * bins + step * steps]) }
            }
        #endif
        default:
            throw Failure(message: "Speech model features have an unexpected type.")
        }

        let encoded = try encoder.encode(melFrames: melFrames)
        let frames = min(encoded, (samples.count + Self.frameSamples - 1) / Self.frameSamples)
        return decoder.decode(encoder.output.contents().bindMemory(to: Float.self, capacity: encoded * SpeechEncoder.width), frames: frames)
    }

    private func value(_ provider: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw Failure(message: "Speech model output \(name) is missing.")
        }
        return array
    }

    private func text(_ tokens: [Int]) -> String {
        tokens.map { vocabulary.indices.contains($0) ? vocabulary[$0] : "" }.joined()
            .replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }

    static func floats(_ pcm: Data) -> [Float] {
        pcm.withUnsafeBytes { raw in
            let count = raw.count / 2
            return (0..<count).map { Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: Int16.self))) / 32_768 }
        }
    }
}

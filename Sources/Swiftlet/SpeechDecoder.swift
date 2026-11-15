import CSwiftlet
import Foundation

/// Greedy TDT decoding for Fluid Speech Pro on the CPU: the 2-layer LSTM prediction network and the joint network,
/// with fp16 weights mapped from decoder.bin (~35 MB, file-backed) and NEON matrix-vector products in fp32.
///
/// Per emitted token: two LSTM steps ([W_ih | W_hh] · [x; h], 2560 × 1280) and the decoder projection. Per visited
/// frame and token: the 8198 × 640 output head, split across the performance cores. Not thread-safe.
final class SpeechDecoder {
    private struct Tensor {
        let pointer: UnsafeRawPointer
        let rows: Int, columns: Int
    }

    static let format = "swiftlet-decoder-1"
    static let blank = 8_192
    private static let hidden = 640
    private static let encoderWidth = 1_024
    private static let outputs = 8_198
    private static let maxSymbolsPerFrame = 10

    private let file: Data
    private let embedding, layer0, bias0, layer1, bias1, encoderWeight, encoderBias, decoderWeight, decoderBias, head, headBias: Tensor
    /// fp32 scratch: x = [input; h] (1280), gates (2560), h and c per layer (2 × 640 each), projections, logits.
    private let scratch: UnsafeMutablePointer<Float>
    private let x, gates, h, c, decoderProjection, encoderProjection, joint, logits: UnsafeMutablePointer<Float>

    init(url: URL) throws {
        file = try Data(contentsOf: url, options: .alwaysMapped)
        let headerLength = file.count > 8 ? file.prefix(8).withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self)) } : 0
        guard headerLength > 0, 8 + headerLength < file.count,
              let header = try? JSONSerialization.jsonObject(with: file.subdata(in: 8..<8 + headerLength)) as? [String: Any],
              header["format"] as? String == Self.format,
              let index = header["tensors"] as? [String: [String: Any]] else {
            throw SpeechEngine.Failure(message: "Speech model file decoder.bin has an unknown format.")
        }
        let start = (8 + headerLength + 16_383) / 16_384 * 16_384
        let base = file.withUnsafeBytes { $0.baseAddress! }  // mapped: stays valid while `file` lives
        let size = file.count
        func tensor(_ name: String, _ rows: Int, _ columns: Int) throws -> Tensor {
            guard let entry = index[name], let offset = entry["offset"] as? Int, let shape = entry["shape"] as? [Int],
                  shape.reduce(1, *) == rows * columns, start + offset + rows * columns * 2 <= size else {
                throw SpeechEngine.Failure(message: "Speech model file decoder.bin is incomplete (\(name)).")
            }
            return Tensor(pointer: base + start + offset, rows: rows, columns: columns)
        }
        let hidden = Self.hidden
        embedding = try tensor("emb", Self.blank + 1, hidden)
        layer0 = try tensor("l0.w", 4 * hidden, 2 * hidden); bias0 = try tensor("l0.b", 4 * hidden, 1)
        layer1 = try tensor("l1.w", 4 * hidden, 2 * hidden); bias1 = try tensor("l1.b", 4 * hidden, 1)
        encoderWeight = try tensor("enc.w", hidden, Self.encoderWidth); encoderBias = try tensor("enc.b", hidden, 1)
        decoderWeight = try tensor("dec.w", hidden, hidden); decoderBias = try tensor("dec.b", hidden, 1)
        head = try tensor("head.w", Self.outputs, hidden); headBias = try tensor("head.b", Self.outputs, 1)

        let counts = [2 * hidden, 4 * hidden, 2 * hidden, 2 * hidden, hidden, hidden, hidden, Self.outputs]
        scratch = .allocate(capacity: counts.reduce(0, +))
        scratch.initialize(repeating: 0, count: counts.reduce(0, +))
        var cursor = scratch
        var pointers: [UnsafeMutablePointer<Float>] = []
        for count in counts { pointers.append(cursor); cursor += count }
        (x, gates, h, c, decoderProjection, encoderProjection, joint, logits) =
            (pointers[0], pointers[1], pointers[2], pointers[3], pointers[4], pointers[5], pointers[6], pointers[7])

        var cores: Int32 = 0, length = MemoryLayout<Int32>.size
        if sysctlbyname("hw.perflevel0.physicalcpu", &cores, &length, nil, 0) != 0 || cores < 1 { cores = 4 }
        swiftlet_pool_start(min(cores, 8))
    }

    deinit { scratch.deallocate() }

    /// `encoded` is fp32 [frames][1024]; returns the token ids (no blanks).
    func decode(_ encoded: UnsafePointer<Float>, frames: Int) -> [Int] {
        h.update(repeating: 0, count: 2 * Self.hidden)
        c.update(repeating: 0, count: 2 * Self.hidden)
        predict(Self.blank)
        var tokens: [Int] = []
        var time = 0, symbols = 0, projected = -1
        while time < frames {
            if projected != time {
                swiftlet_matvec_f16(encoderWeight.pointer, encoderBias.pointer, encoded + time * Self.encoderWidth, encoderProjection,
                                 Int32(Self.encoderWidth), 0, Int32(Self.hidden))
                projected = time
            }
            for i in 0..<Self.hidden { joint[i] = max(encoderProjection[i] + decoderProjection[i], 0) }
            swiftlet_pool_matvec(head.pointer, headBias.pointer, joint, logits, Int32(Self.hidden), Int32(Self.outputs))
            var token = 0, best = -Float.infinity
            for i in 0...Self.blank where logits[i] > best { best = logits[i]; token = i }
            var duration = 0, longest = -Float.infinity  // durations 0 ... 4 frames
            for i in 0..<5 where logits[Self.blank + 1 + i] > longest { longest = logits[Self.blank + 1 + i]; duration = i }
            if token == Self.blank {
                time += max(duration, 1); symbols = 0
                continue
            }
            tokens.append(token)
            predict(token)
            symbols += 1
            // Same stall guard as NeMo: cap zero-duration emissions per frame.
            if duration == 0 && symbols >= Self.maxSymbolsPerFrame { duration = 1 }
            if duration > 0 { symbols = 0 }
            time += duration
        }
        return tokens
    }

    /// Advances the LSTM with `token` (blank = the start state) and updates decoderProjection.
    private func predict(_ token: Int) {
        let hidden = Self.hidden
        #if arch(arm64)  // Float16 doesn't exist on Intel, where the engine refuses to load
        let row = embedding.pointer.assumingMemoryBound(to: Float16.self) + token * hidden
        for i in 0..<hidden { x[i] = Float(row[i]) }
        #endif
        for layer in 0..<2 {
            let state = h + layer * hidden, cell = c + layer * hidden
            (x + hidden).update(from: state, count: hidden)
            let weight = layer == 0 ? layer0 : layer1, bias = layer == 0 ? bias0 : bias1
            swiftlet_pool_matvec(weight.pointer, bias.pointer, x, gates, Int32(2 * hidden), Int32(4 * hidden))
            // Gate rows are ordered input, forget, output, cell.
            for i in 0..<hidden {
                let input = 1 / (1 + exp(-gates[i])), forget = 1 / (1 + exp(-gates[hidden + i]))
                let output = 1 / (1 + exp(-gates[2 * hidden + i])), candidate = tanh(gates[3 * hidden + i])
                cell[i] = forget * cell[i] + input * candidate
                state[i] = output * tanh(cell[i])
                x[i] = state[i]
            }
        }
        swiftlet_matvec_f16(decoderWeight.pointer, decoderBias.pointer, x, decoderProjection, Int32(hidden), 0, Int32(hidden))
    }
}

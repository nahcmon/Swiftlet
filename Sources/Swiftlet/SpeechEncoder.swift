import Foundation
import Metal

/// The Fluid Speech Pro encoder (FastConformer, 24 layers) on the GPU with Swiftlet's own Metal kernels.
///
/// The 4-bit palette weights in encoder.bin are mapped straight into one Metal buffer, so they stay compressed and
/// file-backed (~310 MB clean memory). Only the frames of the actual audio are computed; nothing is padded to 15 s.
/// Not thread-safe: one caller at a time.
final class SpeechEncoder {
    private struct GemmParams { var M, N, K: Int32; var act: Int32; var alpha: Float; var hasBias, hasRes, kLen, partial: Int32 }
    private struct BParams { var M, N, K, lda, ldb, ldc, transB, outHalf, sA, sB, sC: Int32 }
    private struct SubParams { var T0, T1, T2, melStride: Int32 }
    private struct DwParams { var Ti, Fi, To, Fo: Int32 }
    private struct Dw1Params { var T, C, k, act: Int32 }
    private struct Entry { let offset: Int; let lut: Int; let shape: [Int] }
    private typealias Ref = (buffer: MTLBuffer, offset: Int)

    static let format = "swiftlet-encoder-wq2-1"
    /// Mel frames per window (15 s); the encoder output has at most `maxFrames` frames.
    static let maxMel = 1_501
    static let maxFrames = 188
    static let width = 1_024
    private static let layers = 24
    private static let partBytes = 4 << 20

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let weights: MTLBuffer
    private let tensors: [String: Entry]
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let subA, subB, x, h, h2, qu, qv, att, big, s1, s2, pm, posAll, part: MTLBuffer
    /// Input: fp32 log-mel [128][maxMel]. Only the first `frames` columns are read.
    let mel: MTLBuffer
    /// Output: fp32 [frames][1024].
    let output: MTLBuffer

    init(weightsURL: URL) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw SpeechEngine.Failure(message: "Swiftlet needs a Metal GPU.")
        }
        self.device = device
        self.queue = queue

        let descriptor = open(weightsURL.path, O_RDONLY)
        guard descriptor >= 0 else { throw SpeechEngine.Failure(message: "Speech model file encoder.bin can't be opened.") }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_size > 16_384 else {
            throw SpeechEngine.Failure(message: "Speech model file encoder.bin is incomplete.")
        }
        let page = Int(getpagesize())
        let length = (Int(info.st_size) + page - 1) / page * page
        guard let mapped = mmap(nil, length, PROT_READ, MAP_PRIVATE, descriptor, 0), mapped != MAP_FAILED else {
            throw SpeechEngine.Failure(message: "Speech model file encoder.bin can't be mapped.")
        }
        guard let buffer = device.makeBuffer(bytesNoCopy: mapped, length: length, options: .storageModeShared, deallocator: { munmap($0, $1) }) else {
            munmap(mapped, length)
            throw SpeechEngine.Failure(message: "Speech model doesn't fit in GPU memory.")
        }
        weights = buffer
        let headerLength = Int(mapped.loadUnaligned(as: UInt64.self))
        guard headerLength > 0, 8 + headerLength < Int(info.st_size),
              let header = try? JSONSerialization.jsonObject(with: Data(bytes: mapped + 8, count: headerLength)) as? [String: Any],
              header["format"] as? String == Self.format,
              let index = header["tensors"] as? [String: [String: Any]] else {
            throw SpeechEngine.Failure(message: "Speech model file encoder.bin has an unknown format.")
        }
        let start = (8 + headerLength + 16_383) / 16_384 * 16_384
        var tensors: [String: Entry] = [:]
        for (name, value) in index {
            guard let offset = value["offset"] as? Int, let shape = value["shape"] as? [Int] else { continue }
            let lut = value["lut"] as? Int
            let entry = Entry(offset: start + offset, lut: start + (lut ?? 0), shape: shape)
            let bytes = shape.reduce(1, *) * (lut == nil ? 2 : 1) / (lut == nil ? 1 : 2)
            guard entry.offset + bytes <= Int(info.st_size) else {
                throw SpeechEngine.Failure(message: "Speech model file encoder.bin is incomplete.")
            }
            tensors[name] = entry
        }
        self.tensors = tensors
        guard tensors["pos_emb"] != nil, tensors["l\(Self.layers - 1).ff2b.w"] != nil else {
            throw SpeechEngine.Failure(message: "Speech model file encoder.bin is incomplete.")
        }

        let library = try device.makeLibrary(source: SpeechKernels.source, options: nil)
        for name in ["wq2_16x64", "wq3_m8n2_1x4", "gemm_reduce", "bgemm", "layernorm", "sub_conv01", "sub_dw", "dwconv_time",
                     "att_bias", "att_softmax", "to_float"] {
            guard let function = library.makeFunction(name: name) else {
                throw SpeechEngine.Failure(message: "Speech kernel \(name) is missing.")
            }
            pipelines[name] = try device.makeComputePipelineState(function: function)
        }

        func scratch(_ bytes: Int, shared: Bool = false) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: bytes, options: shared ? .storageModeShared : .storageModePrivate) else {
                throw SpeechEngine.Failure(message: "Not enough GPU memory for the speech model.")
            }
            return buffer
        }
        let T = Self.maxFrames, row = T * Self.width * 2
        subA = try scratch(376 * 32 * 256 * 2); subB = try scratch(376 * 32 * 256 * 2)
        x = try scratch(row); h = try scratch(row); h2 = try scratch(row); qu = try scratch(row); qv = try scratch(row); att = try scratch(row)
        big = try scratch(row * 4)
        s1 = try scratch(8 * T * T * 4); s2 = try scratch(8 * T * (2 * T - 1) * 4); pm = try scratch(8 * T * T * 2)
        posAll = try scratch(Self.layers * (2 * T - 1) * Self.width * 2)
        part = try scratch(Self.partBytes)
        mel = try scratch(128 * Self.maxMel * 4, shared: true)
        output = try scratch(T * Self.width * 4, shared: true)

        // Relative-position projections of every layer, once: P = pos_emb · W_posᵀ + b for positions 187 ... -187.
        try run { encoder in
            for layer in 0..<Self.layers {
                try gemm(encoder, input: tensor("pos_emb"), weight: "l\(layer).pos",
                         output: (posAll, layer * (2 * T - 1) * Self.width * 2), M: 2 * T - 1)
            }
        }
    }

    /// Encodes `melFrames` columns of `mel` (1 ... maxMel) and returns the number of output frames in `output`.
    func encode(melFrames T0: Int) throws -> Int {
        precondition(T0 >= 1 && T0 <= Self.maxMel)
        let T1 = (T0 - 1) / 2 + 1, T2 = (T1 - 1) / 2 + 1, t = (T2 - 1) / 2 + 1
        try run { encoder in
            // Subsampling: two stride-2 convolutions over (time, mel), then a third, each followed by a pointwise one.
            var sub = SubParams(T0: Int32(T0), T1: Int32(T1), T2: Int32(T2), melStride: Int32(Self.maxMel))
            try dispatch(encoder, "sub_conv01", [(mel, 0), try tensor("sub.conv0.w"), try tensor("sub.conv0.b"), try tensor("sub.dw1.w"),
                                                 try tensor("sub.dw1.b"), (subA, 0)], &sub, threads: MTLSize(width: 256, height: 32, depth: T2))
            try gemm(encoder, input: (subA, 0), weight: "sub.pw1", output: (subB, 0), M: T2 * 32, act: 2)
            var dw = DwParams(Ti: Int32(T2), Fi: 32, To: Int32(t), Fo: 16)
            try dispatch(encoder, "sub_dw", [(subB, 0), try tensor("sub.dw2.w"), try tensor("sub.dw2.b"), (subA, 0)], &dw,
                         threads: MTLSize(width: 256, height: 16, depth: t))
            try gemm(encoder, input: (subA, 0), weight: "sub.pw2", output: (subB, 0), M: t * 16, act: 2)
            try gemm(encoder, input: (subB, 0), weight: "sub.linear", output: (x, 0), M: t)

            // Insert adapter.
            try layernorm(encoder, x, "ins.ln", h, rows: t)
            try gemm(encoder, input: (h, 0), weight: "ins.down", output: (h2, 0), M: t)
            try depthwise(encoder, h2, "ins.mix", h, T: t, kernel: 15, act: 2)
            try gemm(encoder, input: (h, 0), weight: "ins.up", output: (x, 0), M: t, residual: (x, 0))

            let T = Int32(t)
            for layer in 0..<Self.layers {
                let p = "l\(layer)."
                try layernorm(encoder, x, p + "ln_ff1", h, rows: t)
                try gemm(encoder, input: (h, 0), weight: p + "ff1a", output: (big, 0), M: t, act: 1)
                try gemm(encoder, input: (big, 0), weight: p + "ff1b", output: (x, 0), M: t, residual: (x, 0), alpha: 0.5)

                // Relative-position self-attention, 8 heads of 128.
                try layernorm(encoder, x, p + "ln_att", h, rows: t)
                try gemm(encoder, input: (h, 0), weight: p + "qkv", output: (big, 0), M: t)
                try dispatch(encoder, "att_bias", [(big, 0), try tensor(p + "bias_u"), try tensor(p + "bias_v"), (qu, 0), (qv, 0)],
                             threads: MTLSize(width: Self.width, height: t, depth: 1))
                bgemm(encoder, a: (qu, 0), b: (big, Self.width * 2), c: (s1, 0), half: false,
                      BParams(M: T, N: T, K: 128, lda: 1024, ldb: 3072, ldc: T, transB: 1, outHalf: 0, sA: 128, sB: 128, sC: T * T))
                let positions = layer * (2 * Self.maxFrames - 1) + Self.maxFrames - t
                bgemm(encoder, a: (qv, 0), b: (posAll, positions * Self.width * 2), c: (s2, 0), half: false,
                      BParams(M: T, N: 2 * T - 1, K: 128, lda: 1024, ldb: 1024, ldc: 2 * T - 1, transB: 1, outHalf: 0, sA: 128, sB: 128,
                              sC: T * (2 * T - 1)))
                var frames = T
                try dispatch(encoder, "att_softmax", [(s1, 0), (s2, 0), (pm, 0)], &frames,
                             groups: MTLSize(width: t, height: 1, depth: 1), groupSize: 256)
                bgemm(encoder, a: (pm, 0), b: (big, 2 * Self.width * 2), c: (att, 0), half: true,
                      BParams(M: T, N: 128, K: T, lda: T, ldb: 3072, ldc: 1024, transB: 0, outHalf: 1, sA: T * T, sB: 128, sC: 128))
                try gemm(encoder, input: (att, 0), weight: p + "o", output: (x, 0), M: t, residual: (x, 0))

                try layernorm(encoder, x, p + "ln_conv", h, rows: t)
                try gemm(encoder, input: (h, 0), weight: p + "pw1", output: (h2, 0), M: t, act: 3)
                try depthwise(encoder, h2, p + "dw", h, T: t, kernel: 9, act: 1)
                try gemm(encoder, input: (h, 0), weight: p + "pw2", output: (x, 0), M: t, residual: (x, 0))

                try layernorm(encoder, x, p + "ln_ff2", h, rows: t)
                try gemm(encoder, input: (h, 0), weight: p + "ff2a", output: (big, 0), M: t, act: 1)
                try gemm(encoder, input: (big, 0), weight: p + "ff2b", output: (x, 0), M: t, residual: (x, 0), alpha: 0.5)
                try layernorm(encoder, x, p + "ln_out", x, rows: t)
            }
            try dispatch(encoder, "to_float", [(x, 0), (output, 0)], threads: MTLSize(width: t * Self.width, height: 1, depth: 1))
        }
        return t
    }

    // MARK: - Encoding helpers

    private func run(_ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw SpeechEngine.Failure(message: "The GPU didn't accept the speech model's work.")
        }
        do { try body(encoder) } catch { encoder.endEncoding(); throw error }
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw SpeechEngine.Failure(message: "The speech model failed on the GPU: \(error.localizedDescription)") }
    }

    private func tensor(_ name: String) throws -> Ref {
        guard let entry = tensors[name] else { throw SpeechEngine.Failure(message: "Speech model tensor \(name) is missing.") }
        return (weights, entry.offset)
    }

    private func pipeline(_ encoder: MTLComputeCommandEncoder, _ name: String) {
        encoder.setComputePipelineState(pipelines[name]!)  // every name was compiled in init
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder, _ name: String, _ buffers: [Ref], threads: MTLSize) throws {
        pipeline(encoder, name)
        for (index, ref) in buffers.enumerated() { encoder.setBuffer(ref.buffer, offset: ref.offset, index: index) }
        encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func dispatch<P>(_ encoder: MTLComputeCommandEncoder, _ name: String, _ buffers: [Ref], _ parameters: inout P,
                             threads: MTLSize? = nil, groups: MTLSize? = nil, groupSize: Int = 256) throws {
        pipeline(encoder, name)
        for (index, ref) in buffers.enumerated() { encoder.setBuffer(ref.buffer, offset: ref.offset, index: index) }
        withUnsafeBytes(of: &parameters) { encoder.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
        let size = MTLSize(width: groupSize, height: 1, depth: 1)
        if let groups { encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: size) }
        if let threads { encoder.dispatchThreads(threads, threadsPerThreadgroup: size) }
    }

    /// output = act(alpha · (input · Wᵀ + b) + residual) with 4-bit palette W [N][K].
    private func gemm(_ encoder: MTLComputeCommandEncoder, input: Ref, weight: String, output: Ref, M: Int, act: Int32 = 0,
                      residual: Ref? = nil, alpha: Float = 1) throws {
        guard let entry = tensors[weight + ".w"], entry.shape.count == 2 else {
            throw SpeechEngine.Failure(message: "Speech model tensor \(weight) is missing.")
        }
        let bias = try tensor(weight + ".b")
        let N = entry.shape[0], K = entry.shape[1]
        // Short inputs use 16-row tiles, long ones 64×64. K is split until the grid has ≥ 256 threadgroups
        // (fp32 partial sums, then gemm_reduce applies the epilogue).
        let small = M <= 140
        let rows = small ? 16 : 64, tiles = (N / 64) * ((M + rows - 1) / rows)
        var split = 1
        while tiles * split < 256 && K / (split * 2) >= 256 && split * 2 * M * N * 4 <= Self.partBytes { split *= 2 }
        let target = residual ?? output
        var parameters = GemmParams(M: Int32(M), N: Int32(N), K: Int32(K), act: act, alpha: alpha, hasBias: 1,
                                    hasRes: residual == nil ? 0 : 1, kLen: Int32(K / split), partial: split > 1 ? 1 : 0)
        pipeline(encoder, small ? "wq2_16x64" : "wq3_m8n2_1x4")
        encoder.setBuffer(input.buffer, offset: input.offset, index: 0)
        encoder.setBuffer(weights, offset: entry.offset, index: 1)
        encoder.setBuffer(weights, offset: entry.lut, index: 2)
        encoder.setBuffer(bias.buffer, offset: bias.offset, index: 3)
        encoder.setBuffer(target.buffer, offset: target.offset, index: 4)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 5)
        encoder.setBytes(&parameters, length: MemoryLayout<GemmParams>.stride, index: 6)
        encoder.setBuffer(part, offset: 0, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: (M + rows - 1) / rows, height: N / 64, depth: split),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        guard split > 1 else { return }
        pipeline(encoder, "gemm_reduce")
        encoder.setBuffer(part, offset: 0, index: 0)
        encoder.setBuffer(bias.buffer, offset: bias.offset, index: 1)
        encoder.setBuffer(target.buffer, offset: target.offset, index: 2)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 3)
        encoder.setBytes(&parameters, length: MemoryLayout<GemmParams>.stride, index: 4)
        var splits = Int32(split)
        encoder.setBytes(&splits, length: 4, index: 5)
        encoder.dispatchThreads(MTLSize(width: N / 2, height: M, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    private func layernorm(_ encoder: MTLComputeCommandEncoder, _ input: MTLBuffer, _ prefix: String, _ output: MTLBuffer, rows: Int) throws {
        var channels = Int32(Self.width)
        try dispatch(encoder, "layernorm", [(input, 0), try tensor(prefix + ".g"), try tensor(prefix + ".b"), (output, 0)], &channels,
                     groups: MTLSize(width: rows, height: 1, depth: 1))
    }

    private func depthwise(_ encoder: MTLComputeCommandEncoder, _ input: MTLBuffer, _ prefix: String, _ output: MTLBuffer,
                           T: Int, kernel: Int, act: Int32) throws {
        var parameters = Dw1Params(T: Int32(T), C: Int32(Self.width), k: Int32(kernel), act: act)
        try dispatch(encoder, "dwconv_time", [(input, 0), try tensor(prefix + ".w"), try tensor(prefix + ".b"), (output, 0)], &parameters,
                     threads: MTLSize(width: Self.width, height: T, depth: 1))
    }

    /// Batched (8 heads) fp16 GEMM; `half` writes fp16 output (index 3) instead of fp32 (index 2).
    private func bgemm(_ encoder: MTLComputeCommandEncoder, a: Ref, b: Ref, c: Ref, half: Bool, _ parameters: BParams) {
        var parameters = parameters
        pipeline(encoder, "bgemm")
        encoder.setBuffer(a.buffer, offset: a.offset, index: 0)
        encoder.setBuffer(b.buffer, offset: b.offset, index: 1)
        encoder.setBuffer(c.buffer, offset: c.offset, index: half ? 3 : 2)
        encoder.setBuffer(c.buffer, offset: c.offset, index: half ? 2 : 3)
        encoder.setBytes(&parameters, length: MemoryLayout<BParams>.stride, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: (Int(parameters.N) + 31) / 32, height: (Int(parameters.M) + 31) / 32, depth: 8),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}

import CryptoKit
import Foundation

/// Downloads the 4-bit Fluid Speech Pro weights for Swiftlet from Hugging Face and checks every file's SHA-256.
public enum ModelDownloader {
    public static let repository = "nahcmon/fluid-speech-pro-swiftlet-4bit"
    /// The Hugging Face commit these hashes belong to.
    public static let revision = "c431a65b9914e8305f185613b132eda8276cef32"

    static let files: [(path: String, bytes: Int, sha256: String)] = [
        ("encoder.bin", 322_330_624, "b70247b41b49852e119194360fc19de4b46a4ff030acd0d021dfb05930333182"),
        ("decoder.bin", 36_263_424, "eea8e5016eaa32408758beb6a0189b09a64028094485e2953fec49a4dbe254df"),
        ("parakeet_v3_vocab.json", 151_122, "7ec60e05f1b24480736ec0eed40900f4626bce1fa9a60fd700ec7e2a59198735"),
        ("Preprocessor.mlmodelc/coremldata.bin", 429, "38f130eac9ffde8c684043d76e3eecdd3b6f9564d9efa59f7a87a34a272a4901"),
        ("Preprocessor.mlmodelc/model.mil", 11_891, "71d4ad016334b3ad73ea7b69b6813feefe3b81f59e0b5e924f90f088cb02a7c9"),
        ("Preprocessor.mlmodelc/weights/weight.bin", 2_150_580, "44180071e400951080786feb46452bffc4fb26363a703655ebadb760c8ad4030"),
        ("Preprocessor.mlmodelc/metadata.json", 2_606, "c25d98c19a181b7fa65ee50c34f5066f67172c22f0bf116a42b59443ee4bc37d"),
        ("Preprocessor.mlmodelc/analytics/coremldata.bin", 243, "886396da81b6b29d60d79d903bd5d1f00077a7a4323f40cd2e9214d4205d8e6b"),
    ]

    /// Total download size in bytes (about 361 MB).
    public static var totalBytes: Int { files.reduce(0) { $0 + $1.bytes } }

    /// `~/Library/Application Support/Swiftlet/fluid-speech-pro-4bit`
    public static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Swiftlet/fluid-speech-pro-4bit", isDirectory: true)
    }

    /// Whether every model file in `directory` is present with the expected size.
    public static func isDownloaded(at directory: URL = defaultDirectory) -> Bool {
        files.allSatisfy { size(directory.appendingPathComponent($0.path)) == $0.bytes }
    }

    /// Downloads whatever is missing into `directory` and returns it. `progress` gets bytes done and total.
    @discardableResult
    public static func download(to directory: URL = defaultDirectory,
                                progress: (@Sendable (Int, Int) -> Void)? = nil) async throws -> URL {
        let manager = FileManager.default
        var done = 0
        for file in files {
            let target = directory.appendingPathComponent(file.path)
            if size(target) == file.bytes { done += file.bytes; progress?(done, totalBytes); continue }
            let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(file.path)")!
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw SpeechEngine.Failure(message: "Download of \(file.path) failed (\((response as? HTTPURLResponse)?.statusCode ?? 0)).")
            }
            guard try sha256(temporary) == file.sha256 else {
                throw SpeechEngine.Failure(message: "Downloaded \(file.path) doesn't match its checksum.")
            }
            try manager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            if manager.fileExists(atPath: target.path) { try manager.removeItem(at: target) }
            try manager.moveItem(at: temporary, to: target)
            done += file.bytes
            progress?(done, totalBytes)
        }
        return directory
    }

    private static func size(_ url: URL) -> Int {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.intValue ?? -1
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

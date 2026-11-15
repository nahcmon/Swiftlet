import Foundation
import Swiftlet

// swiftlet download [model dir]
// swiftlet transcribe [--model <dir>] [--time] <audio>...

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first, ["download", "transcribe"].contains(command) else {
    fail("usage: swiftlet download [model dir]\n       swiftlet transcribe [--model <dir>] [--time] <audio>...")
}
args.removeFirst()

if command == "download" {
    let directory = args.first.map { URL(fileURLWithPath: $0) } ?? ModelDownloader.defaultDirectory
    do {
        let started = Date()
        try await ModelDownloader.download(to: directory) { done, total in
            FileHandle.standardError.write(String(format: "\r%.0f / %.0f MB", Double(done) / 1e6, Double(total) / 1e6).data(using: .utf8)!)
        }
        FileHandle.standardError.write(String(format: "\nModel ready in %@ (%.0f s)\n", directory.path, Date().timeIntervalSince(started)).data(using: .utf8)!)
    } catch { fail("swiftlet: \(error.localizedDescription)") }
    exit(0)
}

var model = ModelDownloader.defaultDirectory
var timing = false
var files: [String] = []
while !args.isEmpty {
    let argument = args.removeFirst()
    switch argument {
    case "--model": guard !args.isEmpty else { fail("--model needs a folder") }; model = URL(fileURLWithPath: args.removeFirst())
    case "--time": timing = true
    default: files.append(argument)
    }
}
guard !files.isEmpty else { fail("swiftlet transcribe: no audio files given") }
guard ModelDownloader.isDownloaded(at: model) || FileManager.default.fileExists(atPath: model.appendingPathComponent("encoder.bin").path) else {
    fail("No model in \(model.path). Run `swiftlet download` first.")
}
do {
    let loading = Date()
    let engine = try SpeechEngine(directory: model)
    if timing { FileHandle.standardError.write(String(format: "load %.0f ms\n", Date().timeIntervalSince(loading) * 1000).data(using: .utf8)!) }
    for path in files {
        let started = Date()
        let text = try engine.transcribe(contentsOf: URL(fileURLWithPath: path))
        print(text)
        if timing { FileHandle.standardError.write(String(format: "%@: %.0f ms\n", path, Date().timeIntervalSince(started) * 1000).data(using: .utf8)!) }
    }
} catch { fail("swiftlet: \(error.localizedDescription)") }

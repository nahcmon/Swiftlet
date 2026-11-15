import AVFoundation
import Foundation

enum AudioFile {
    /// Any audio file AVFoundation can read, as 16 kHz mono Int16 little-endian PCM.
    static func pcm16k(_ url: URL) throws -> Data {
        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: Double(SpeechEngine.sampleRate), channels: 1, interleaved: true)!
        guard let converter = AVAudioConverter(from: file.processingFormat, to: format),
              let source = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 8192),
              let destination = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8192) else {
            throw SpeechEngine.Failure(message: "Cannot convert \(url.lastPathComponent) to 16 kHz mono.")
        }
        var pcm = Data()
        while true {
            var error: NSError?
            let status = converter.convert(to: destination, error: &error) { _, input in
                guard file.framePosition < file.length,
                      (try? file.read(into: source, frameCount: AVAudioFrameCount(min(8192, file.length - file.framePosition)))) != nil,
                      source.frameLength > 0 else { input.pointee = .endOfStream; return nil }
                input.pointee = .haveData; return source
            }
            if let error { throw error }
            if let samples = destination.int16ChannelData?[0], destination.frameLength > 0 {
                pcm.append(Data(bytes: samples, count: Int(destination.frameLength) * 2))
            }
            if status == .endOfStream || status == .error { break }
        }
        return pcm
    }
}

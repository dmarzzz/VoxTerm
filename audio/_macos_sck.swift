// VOXTERM — System Audio Capture Helper
// Captures all desktop audio via ScreenCaptureKit, resamples to 16kHz mono float32,
// and streams raw PCM to stdout. Designed to be launched as a subprocess by VoxTerm.
//
// Exit codes: 0 = clean shutdown, 1 = permission denied, 2 = other error
// Compile: swiftc -O -o sck-helper _macos_sck.swift

import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia

// MARK: - Stream Output Delegate

class AudioOutputHandler: NSObject, SCStreamOutput, SCStreamDelegate {
    private let stdout = FileHandle.standardOutput
    private var converter: AVAudioConverter?
    private var inputFormat: AVAudioFormat?
    private let outputFormat: AVAudioFormat
    private let lock = NSLock()

    override init() {
        outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!
        super.init()
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio else { return }
        guard CMSampleBufferIsValid(sampleBuffer) else { return }
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer) else { return }
        guard let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc)?.pointee else { return }

        lock.lock()
        defer { lock.unlock() }

        // Lazily create converter on first buffer (now we know the actual input format)
        if converter == nil {
            FileHandle.standardError.write("sck: first audio buffer — \(asbd.mSampleRate)Hz, \(asbd.mChannelsPerFrame)ch\n".data(using: .utf8)!)
            guard let inFmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: asbd.mSampleRate,
                channels: AVAudioChannelCount(asbd.mChannelsPerFrame),
                interleaved: false
            ) else { return }
            inputFormat = inFmt
            converter = AVAudioConverter(from: inFmt, to: outputFormat)
            if converter == nil {
                FileHandle.standardError.write("error: failed to create audio converter\n".data(using: .utf8)!)
                return
            }
        }

        guard let converter = converter, let inputFormat = inputFormat else { return }

        let numSamples = CMSampleBufferGetNumSamples(sampleBuffer)
        guard numSamples > 0 else { return }

        // Extract audio buffer list from CMSampleBuffer
        var blockBuffer: CMBlockBuffer?
        let ablSize = MemoryLayout<AudioBufferList>.size
        var abl = AudioBufferList()
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, bufferListSizeNeededOut: nil,
            bufferListOut: &abl, bufferListSize: ablSize,
            blockBufferAllocator: nil, blockBufferMemoryAllocator: nil,
            flags: 0, blockBufferOut: &blockBuffer
        )
        guard status == noErr else { return }

        // Create input PCM buffer
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat,
                                                  frameCapacity: AVAudioFrameCount(numSamples)) else { return }
        inputBuffer.frameLength = AVAudioFrameCount(numSamples)

        // Copy audio data into input buffer
        let channelCount = Int(inputFormat.channelCount)
        if inputFormat.isInterleaved {
            let src = abl.mBuffers.mData
            let dst = inputBuffer.audioBufferList.pointee.mBuffers.mData
            if let src = src, let dst = dst {
                memcpy(dst, src, Int(abl.mBuffers.mDataByteSize))
            }
        } else {
            // Non-interleaved: copy each channel
            let bufferListPtr = UnsafeMutableAudioBufferListPointer(inputBuffer.mutableAudioBufferList)
            let srcData = abl.mBuffers.mData
            let bytesPerFrame = Int(asbd.mBytesPerFrame)
            let totalBytes = numSamples * bytesPerFrame

            if channelCount == 1 {
                if let src = srcData, let dst = bufferListPtr[0].mData {
                    memcpy(dst, src, min(totalBytes, Int(bufferListPtr[0].mDataByteSize)))
                }
            } else {
                // For interleaved source → non-interleaved destination, deinterleave
                if let src = srcData {
                    let srcPtr = src.assumingMemoryBound(to: Float.self)
                    for ch in 0..<min(channelCount, bufferListPtr.count) {
                        if let dst = bufferListPtr[ch].mData?.assumingMemoryBound(to: Float.self) {
                            for frame in 0..<numSamples {
                                dst[frame] = srcPtr[frame * channelCount + ch]
                            }
                        }
                    }
                }
            }
        }

        // Resample to 16kHz mono
        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let outputFrameCapacity = AVAudioFrameCount(Double(numSamples) * ratio + 1)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat,
                                                   frameCapacity: outputFrameCapacity) else { return }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { packetCount, statusPtr in
            statusPtr.pointee = .haveData
            return inputBuffer
        }
        if let error = error {
            FileHandle.standardError.write("convert error: \(error.localizedDescription)\n".data(using: .utf8)!)
            return
        }

        guard outputBuffer.frameLength > 0,
              let channelData = outputBuffer.floatChannelData?[0] else { return }

        // Write raw float32 PCM to stdout
        let byteCount = Int(outputBuffer.frameLength) * MemoryLayout<Float>.size
        let data = Data(bytes: channelData, count: byteCount)
        stdout.write(data)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("stream stopped: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(0)
    }
}

// MARK: - Shutdown

// Global reference so signal handlers can tear down the stream
var activeStream: SCStream?

func shutdownAndExit(code: Int32) -> Never {
    if let stream = activeStream {
        activeStream = nil
        let sem = DispatchSemaphore(value: 0)
        stream.stopCapture { _ in sem.signal() }
        // Give CoreAudio a moment to release the audio tap
        _ = sem.wait(timeout: .now() + 2.0)
    }
    exit(code)
}

// MARK: - Main

func run() {
    // Check macOS version (ScreenCaptureKit requires 12.3+)
    let version = ProcessInfo.processInfo.operatingSystemVersion
    guard version.majorVersion >= 13 || (version.majorVersion == 12 && version.minorVersion >= 3) else {
        FileHandle.standardError.write("error: ScreenCaptureKit requires macOS 12.3+\n".data(using: .utf8)!)
        exit(2)
    }

    let semaphore = DispatchSemaphore(value: 0)
    var shareableContent: SCShareableContent?
    var contentError: Error?

    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: false) { content, error in
        shareableContent = content
        contentError = error
        semaphore.signal()
    }
    semaphore.wait()

    if let error = contentError {
        let msg = error.localizedDescription.lowercased()
        if msg.contains("permission") || msg.contains("denied") || msg.contains("declined") || msg.contains("tcc") {
            FileHandle.standardError.write("error: Screen Recording permission denied. Grant access in System Settings > Privacy & Security > Screen Recording\n".data(using: .utf8)!)
            exit(1)
        }
        FileHandle.standardError.write("error: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(2)
    }

    guard let content = shareableContent, let display = content.displays.first else {
        FileHandle.standardError.write("error: no displays found\n".data(using: .utf8)!)
        exit(2)
    }

    // Configure audio-only capture
    let config = SCStreamConfiguration()
    config.capturesAudio = true
    config.excludesCurrentProcessAudio = true
    config.channelCount = 2
    config.sampleRate = 48000
    // Minimal video (SCK requires non-zero dimensions)
    config.width = 2
    config.height = 2
    config.minimumFrameInterval = CMTime(value: 1, timescale: 1) // 1 FPS minimum
    config.showsCursor = false

    // Filter: capture all system audio except our own process.
    // Using excludingApplications is more inclusive than including — it captures
    // audio from all sources including Chrome helper/renderer processes.
    let ownPID = ProcessInfo.processInfo.processIdentifier
    let selfApp = content.applications.filter { $0.processID == ownPID }
    let filter = SCContentFilter(display: display, excludingApplications: selfApp, exceptingWindows: [])

    // Log captured app count for diagnostics
    let appCount = content.applications.count - selfApp.count
    FileHandle.standardError.write("sck: capturing audio from \(appCount) apps\n".data(using: .utf8)!)

    let handler = AudioOutputHandler()

    let stream = SCStream(filter: filter, configuration: config, delegate: handler)
    activeStream = stream

    let streamQueue = DispatchQueue(label: "com.voxterm.sck-audio", qos: .userInteractive)
    do {
        try stream.addStreamOutput(handler, type: .audio, sampleHandlerQueue: streamQueue)
    } catch {
        FileHandle.standardError.write("error: failed to add stream output: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(2)
    }

    let startSemaphore = DispatchSemaphore(value: 0)
    var startError: Error?

    stream.startCapture { error in
        startError = error
        startSemaphore.signal()
    }
    startSemaphore.wait()

    if let error = startError {
        let msg = error.localizedDescription.lowercased()
        if msg.contains("permission") || msg.contains("denied") || msg.contains("declined") || msg.contains("tcc") {
            FileHandle.standardError.write("error: Screen Recording permission denied. Grant access in System Settings > Privacy & Security > Screen Recording\n".data(using: .utf8)!)
            exit(1)
        }
        FileHandle.standardError.write("error: capture failed: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(2)
    }

    // Signal handlers: stop the stream before exiting so CoreAudio releases the tap
    signal(SIGTERM) { _ in shutdownAndExit(code: 0) }
    signal(SIGPIPE) { _ in shutdownAndExit(code: 0) }

    // Monitor stdin for EOF (parent process died)
    DispatchQueue.global(qos: .utility).async {
        while true {
            let data = FileHandle.standardInput.availableData
            if data.isEmpty {
                shutdownAndExit(code: 0)
            }
        }
    }

    // Run forever until signal or stdin EOF
    RunLoop.main.run()
}

run()

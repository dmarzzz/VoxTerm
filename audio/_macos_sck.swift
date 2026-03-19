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
    private let lock = NSLock()
    private var loggedFirst = false

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio else { return }
        guard CMSampleBufferIsValid(sampleBuffer) else { return }

        lock.lock()
        defer { lock.unlock() }

        if !loggedFirst {
            if let desc = CMSampleBufferGetFormatDescription(sampleBuffer),
               let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(desc)?.pointee {
                FileHandle.standardError.write("sck: audio — \(asbd.mSampleRate)Hz, \(asbd.mChannelsPerFrame)ch\n".data(using: .utf8)!)
            }
            loggedFirst = true
        }

        // Extract raw PCM from the sample buffer and write to stdout
        guard let blockBuf = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        let length = CMBlockBufferGetDataLength(blockBuf)
        guard length > 0 else { return }

        var dataPointer: UnsafeMutablePointer<Int8>?
        var lengthAtOffset: Int = 0
        let status = CMBlockBufferGetDataPointer(blockBuf, atOffset: 0, lengthAtOffsetOut: &lengthAtOffset, totalLengthOut: nil, dataPointerOut: &dataPointer)
        guard status == noErr, let ptr = dataPointer else { return }

        stdout.write(Data(bytes: ptr, count: lengthAtOffset))
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
    config.channelCount = 1
    config.sampleRate = 16000
    // Minimal video (SCK requires non-zero dimensions)
    config.width = 2
    config.height = 2
    config.minimumFrameInterval = CMTime(value: 1, timescale: 1) // 1 FPS minimum
    config.showsCursor = false

    // Filter: capture entire display audio.
    // With BlackHole multi-output device as default, audio flows through
    // a non-Bluetooth path that SCK can tap.
    let filter = SCContentFilter(display: display, excludingWindows: [])
    FileHandle.standardError.write("sck: display filter active\n".data(using: .utf8)!)

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

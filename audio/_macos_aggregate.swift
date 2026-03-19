// VOXTERM — Multi-Output Device Manager
// Creates/destroys CoreAudio aggregate (multi-output) devices for Bluetooth + BlackHole routing.
//
// Usage:
//   aggregate-helper create           — create multi-output device, print JSON result
//   aggregate-helper destroy <uid>    — destroy aggregate device by UID
//   aggregate-helper list             — list audio devices as JSON
//
// Exit codes: 0 = success, 1 = error

import Foundation
import CoreAudio

// MARK: - Audio Device Helpers

struct AudioDeviceInfo {
    let id: AudioDeviceID
    let uid: String
    let name: String
    let isOutput: Bool
    let isInput: Bool
    let transportType: UInt32
}

func getDevices() -> [AudioDeviceInfo] {
    var propSize: UInt32 = 0
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &propSize)
    let count = Int(propSize) / MemoryLayout<AudioDeviceID>.size
    var deviceIDs = [AudioDeviceID](repeating: 0, count: count)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &propSize, &deviceIDs)

    var devices: [AudioDeviceInfo] = []
    for deviceID in deviceIDs {
        guard let uid = getStringProperty(deviceID, selector: kAudioDevicePropertyDeviceUID),
              let name = getStringProperty(deviceID, selector: kAudioObjectPropertyName) else {
            continue
        }

        let hasOutput = hasStreams(deviceID, scope: kAudioDevicePropertyScopeOutput)
        let hasInput = hasStreams(deviceID, scope: kAudioDevicePropertyScopeInput)
        let transport = getTransportType(deviceID)

        devices.append(AudioDeviceInfo(
            id: deviceID, uid: uid, name: name,
            isOutput: hasOutput, isInput: hasInput,
            transportType: transport
        ))
    }
    return devices
}

func getStringProperty(_ deviceID: AudioDeviceID, selector: AudioObjectPropertySelector) -> String? {
    var addr = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var value: CFString = "" as CFString
    var size = UInt32(MemoryLayout<CFString>.size)
    let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &value)
    return status == noErr ? value as String : nil
}

func hasStreams(_ deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> Bool {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreams,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size)
    return size > 0
}

func getTransportType(_ deviceID: AudioDeviceID) -> UInt32 {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyTransportType,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var transport: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &transport)
    return transport
}

func getDefaultOutputUID() -> String? {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var deviceID: AudioDeviceID = 0
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &deviceID)
    guard status == noErr else { return nil }
    return getStringProperty(deviceID, selector: kAudioDevicePropertyDeviceUID)
}

func setDefaultOutputDevice(_ deviceID: AudioDeviceID) -> Bool {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var id = deviceID
    let status = AudioObjectSetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil,
        UInt32(MemoryLayout<AudioDeviceID>.size), &id
    )
    return status == noErr
}

// MARK: - Aggregate Device Management

let kVoxTermAggregateUID = "com.voxterm.multi-output"
let kVoxTermAggregateName = "VoxTerm Multi-Output"

func createMultiOutputDevice(outputUID: String, blackholeUID: String) -> (AudioDeviceID, String)? {
    let subDevices: [[String: Any]] = [
        [kAudioSubDeviceUIDKey as String: outputUID],
        [kAudioSubDeviceUIDKey as String: blackholeUID],
    ]

    let desc: [String: Any] = [
        kAudioAggregateDeviceNameKey as String: kVoxTermAggregateName,
        kAudioAggregateDeviceUIDKey as String: kVoxTermAggregateUID,
        kAudioAggregateDeviceSubDeviceListKey as String: subDevices,
        kAudioAggregateDeviceIsStackedKey as String: 0 as UInt32,  // multi-output, not stacked
    ]

    var deviceID: AudioDeviceID = 0
    let status = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &deviceID)
    if status != noErr {
        return nil
    }

    // Enable drift correction on the BlackHole sub-device (second device)
    enableDriftCorrection(aggregateID: deviceID, subDeviceUID: blackholeUID)

    return (deviceID, kVoxTermAggregateUID)
}

func enableDriftCorrection(aggregateID: AudioDeviceID, subDeviceUID: String) {
    // Get sub-device list
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioAggregateDevicePropertyActiveSubDeviceList,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var propSize: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(aggregateID, &addr, 0, nil, &propSize) == noErr else { return }

    let count = Int(propSize) / MemoryLayout<AudioDeviceID>.size
    var subDeviceIDs = [AudioDeviceID](repeating: 0, count: count)
    guard AudioObjectGetPropertyData(aggregateID, &addr, 0, nil, &propSize, &subDeviceIDs) == noErr else { return }

    // Find the BlackHole sub-device and enable drift correction
    for subID in subDeviceIDs {
        guard let uid = getStringProperty(subID, selector: kAudioDevicePropertyDeviceUID) else { continue }
        if uid == subDeviceUID {
            var driftAddr = AudioObjectPropertyAddress(
                mSelector: kAudioSubDevicePropertyDriftCompensation,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var drift: UInt32 = 1
            AudioObjectSetPropertyData(subID, &driftAddr, 0, nil, UInt32(MemoryLayout<UInt32>.size), &drift)
            break
        }
    }
}

func destroyAggregateDevice(uid: String) -> Bool {
    let devices = getDevices()
    for dev in devices {
        if dev.uid == uid {
            let status = AudioHardwareDestroyAggregateDevice(dev.id)
            return status == noErr
        }
    }
    return false
}

// MARK: - Commands

func cmdList() {
    let devices = getDevices()
    let defaultUID = getDefaultOutputUID() ?? ""
    var entries: [[String: Any]] = []
    for d in devices {
        var entry: [String: Any] = [
            "id": d.id,
            "uid": d.uid,
            "name": d.name,
            "is_output": d.isOutput,
            "is_input": d.isInput,
            "transport_type": d.transportType,
            "is_default_output": d.uid == defaultUID,
        ]
        // 0x626C7565 = 'blue' (Bluetooth transport)
        entry["is_bluetooth"] = d.transportType == 0x626C7565
        entries.append(entry)
    }
    if let json = try? JSONSerialization.data(withJSONObject: entries, options: .prettyPrinted),
       let str = String(data: json, encoding: .utf8) {
        print(str)
    }
}

func cmdCreate() {
    let devices = getDevices()
    let defaultUID = getDefaultOutputUID()

    // Find the current default output device
    guard let currentOutputUID = defaultUID,
          let currentOutput = devices.first(where: { $0.uid == currentOutputUID && $0.isOutput }) else {
        FileHandle.standardError.write("error: could not find default output device\n".data(using: .utf8)!)
        exit(1)
    }

    // Find BlackHole
    guard let blackhole = devices.first(where: { $0.name.lowercased().contains("blackhole") && $0.isOutput }) else {
        FileHandle.standardError.write("error: BlackHole not found — install with: brew install blackhole-2ch\n".data(using: .utf8)!)
        exit(1)
    }

    // Check if our aggregate device already exists
    if let existing = devices.first(where: { $0.uid == kVoxTermAggregateUID }) {
        // Already exists — just set it as default and return its info
        _ = setDefaultOutputDevice(existing.id)
        let result: [String: Any] = [
            "status": "exists",
            "aggregate_uid": kVoxTermAggregateUID,
            "aggregate_id": existing.id,
            "original_output_uid": currentOutputUID,
            "original_output_name": currentOutput.name,
        ]
        if let json = try? JSONSerialization.data(withJSONObject: result, options: []),
           let str = String(data: json, encoding: .utf8) {
            print(str)
        }
        return
    }

    // Create the multi-output device
    guard let (deviceID, uid) = createMultiOutputDevice(
        outputUID: currentOutputUID,
        blackholeUID: blackhole.uid
    ) else {
        FileHandle.standardError.write("error: failed to create multi-output device\n".data(using: .utf8)!)
        exit(1)
    }

    // Set it as default output
    _ = setDefaultOutputDevice(deviceID)

    let result: [String: Any] = [
        "status": "created",
        "aggregate_uid": uid,
        "aggregate_id": deviceID,
        "original_output_uid": currentOutputUID,
        "original_output_name": currentOutput.name,
        "blackhole_name": blackhole.name,
    ]
    if let json = try? JSONSerialization.data(withJSONObject: result, options: []),
       let str = String(data: json, encoding: .utf8) {
        print(str)
    }
}

func cmdDestroy(_ uid: String) {
    // First restore the original default output if we can find a non-aggregate output device
    let devices = getDevices()
    let defaultUID = getDefaultOutputUID()

    if defaultUID == uid {
        // We're about to destroy the current default — switch to something else first
        if let fallback = devices.first(where: {
            $0.isOutput && $0.uid != uid && !$0.name.lowercased().contains("blackhole")
        }) {
            _ = setDefaultOutputDevice(fallback.id)
        }
    }

    if destroyAggregateDevice(uid: uid) {
        print("{\"status\": \"destroyed\"}")
    } else {
        FileHandle.standardError.write("error: could not destroy device \(uid)\n".data(using: .utf8)!)
        exit(1)
    }
}

// MARK: - Main

let args = CommandLine.arguments
if args.count < 2 {
    FileHandle.standardError.write("usage: aggregate-helper [create|destroy <uid>|list]\n".data(using: .utf8)!)
    exit(1)
}

switch args[1] {
case "list":
    cmdList()
case "create":
    cmdCreate()
case "destroy":
    if args.count < 3 {
        FileHandle.standardError.write("usage: aggregate-helper destroy <uid>\n".data(using: .utf8)!)
        exit(1)
    }
    cmdDestroy(args[2])
default:
    FileHandle.standardError.write("unknown command: \(args[1])\n".data(using: .utf8)!)
    exit(1)
}

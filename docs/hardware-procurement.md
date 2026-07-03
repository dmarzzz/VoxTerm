# Hardware Procurement Checklist

This document turns issue #94 into a purchasing packet. It does not place the
orders; it defines the bill of materials, acceptance fields, and receipt trail
needed for the always-on room rollout.

## Target Quantity

Buy or allocate five complete room kits:

| Kit | Device | Mic | Required accessories | Receipt | Room |
|---|---|---|---|---|---|
| voxterm-room-01 | TBD | TBD | power, network, mic cable, labels | TBD | TBD |
| voxterm-room-02 | TBD | TBD | power, network, mic cable, labels | TBD | TBD |
| voxterm-room-03 | TBD | TBD | power, network, mic cable, labels | TBD | TBD |
| voxterm-room-04 | TBD | TBD | power, network, mic cable, labels | TBD | TBD |
| voxterm-room-05 | TBD | TBD | power, network, mic cable, labels | TBD | TBD |

## Device Recommendation

Preferred device: current Apple Mac mini with Apple silicon, at least 16 GB
unified memory, and at least 512 GB storage if budget allows.

Source links:

- Apple Mac mini product page: <https://www.apple.com/mac-mini/>
- Apple Mac mini technical specifications: <https://www.apple.com/mac-mini/specs/>
- Apple Mac mini buy page: <https://www.apple.com/shop/buy-mac/mac-mini>

Why this fits:

- Small enough to mount or hide in a room.
- Current Apple silicon has enough local compute headroom for VoxTerm's
  on-device ASR paths and GUI.
- Built-in Ethernet keeps the room server stable on LAN.
- Front and rear ports reduce adapter friction for a handheld USB mic.

Fallback device: an existing laptop that can stay plugged in, auto-login to a
dedicated room account, expose the GUI server on LAN, and keep microphone
permissions stable after reboot.

Reject devices that:

- cannot run Python 3.12+,
- lack reliable USB audio input,
- cannot stay powered continuously,
- cannot be labeled and assigned to a room,
- require personal accounts for daily operation.

## Microphone Recommendation

Preferred mic class: handheld dynamic USB or USB/XLR mic. Dynamic handheld mics
are more forgiving in rooms than condenser desktop mics, and USB lets each room
kit work without a separate audio interface.

Good candidates to quote:

- Audio-Technica ATR2100x-USB: <https://www.audio-technica.com/en-us/atr2100x-usb>
- Samson Q2U: <https://samsontech.com/products/microphones/usb-microphones/q2u/>
- Rode PodMic USB: <https://rode.com/en-us/products/podmic-usb>
- Shure MV7+: <https://www.shure.com/en-US/products/microphones/mv7>

Selection rules:

- Prefer USB-C direct connection where possible.
- Prefer dynamic/cardioid capsules over room-sensitive condensers.
- Prefer a form factor people can pass around or place close to the speaker.
- Buy a windscreen/pop filter per mic.
- Buy a spare cable per two rooms.
- Avoid wireless-only mics unless the receiver appears as a normal USB audio
  interface and can stay paired after reboot.

Recommended baseline: five ATR2100x-USB or Samson Q2U kits, plus spare USB-C
cables and windscreens. Use Rode PodMic USB or Shure MV7+ only if budget and
mounting are less constrained; they are stronger desk/broadcast mics but less
natural to pass around.

## Accessories Per Kit

- Power cable and surge-protected outlet access.
- Ethernet cable, or documented Wi-Fi credentials if Ethernet is impossible.
- USB-C or USB-A cable that matches the selected mic and device ports.
- Adhesive label for device name, room, and support contact.
- Small stand, clip, or mic holder.
- Optional: powered USB-C hub if the room needs keyboard, mouse, mic, and
  storage attached at once.

## Receipt Trail

Record this for every purchase:

| Field | Required value |
|---|---|
| Purchaser | name/account used |
| Vendor | Apple, manufacturer, retailer, or internal asset pool |
| Date ordered | YYYY-MM-DD |
| Item | exact product name and configuration |
| Quantity | count |
| Unit price | quoted at purchase time |
| Taxes/shipping | quoted at purchase time |
| Receipt path/link | expense-system URL or stored PDF path |
| Assigned kit | voxterm-room-01 through voxterm-room-05 |
| Serial/asset tag | after receiving |

## Acceptance Checklist For Issue #94

- Five room devices are ordered or allocated.
- Five passable handheld room mics are ordered or allocated.
- Cables, labels, and required adapters are included.
- Receipts are filed and linked in the receipt trail.
- Each kit has a device label before handoff to deployment.
- Procurement notes are handed to the #95 deployment owner.

## Hand-Off To Deployment

After hardware arrives, hand the kits to the #95 deployment owner for imaging,
service setup, kiosk launch, room assignment, token handling, and operator
checks.

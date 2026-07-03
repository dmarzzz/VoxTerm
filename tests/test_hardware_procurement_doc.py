from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/hardware-procurement.md"


def test_hardware_procurement_doc_tracks_five_room_kits():
    text = DOC.read_text(encoding="utf-8")

    for index in range(1, 6):
        assert f"voxterm-room-{index:02d}" in text


def test_hardware_procurement_doc_has_current_source_links():
    text = DOC.read_text(encoding="utf-8")

    for url in [
        "https://www.apple.com/mac-mini/",
        "https://www.apple.com/mac-mini/specs/",
        "https://www.apple.com/shop/buy-mac/mac-mini",
        "https://www.audio-technica.com/en-us/atr2100x-usb",
        "https://samsontech.com/products/microphones/usb-microphones/q2u/",
        "https://rode.com/en-us/products/podmic-usb",
        "https://www.shure.com/en-US/products/microphones/mv7",
    ]:
        assert url in text


def test_hardware_procurement_doc_covers_receipts_and_acceptance():
    text = DOC.read_text(encoding="utf-8")

    for phrase in [
        "Receipt path/link",
        "Unit price",
        "Taxes/shipping",
        "Serial/asset tag",
        "Five room devices are ordered or allocated.",
        "Five passable handheld room mics are ordered or allocated.",
        "Receipts are filed and linked in the receipt trail.",
    ]:
        assert phrase in text

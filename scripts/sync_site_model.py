"""Copy the exported model into the website so the demo can run it in the browser.

Why: the website's live demo scores audio on the visitor's own device with
onnxruntime-web. It needs the small (8-bit) model file and the mel bands. The
bands are rewritten as raw 32-bit numbers because a browser cannot read .npy.

Writes to website/public/model/:
  audio_eot.int8.onnx   the same file the Python package ships
  mel_filters.bin       201 x 80 float32, little-endian, row by row
  model.json            sizes the browser code needs
"""

import json
import shutil

import numpy as np

from indic_eot.config import REPO_ROOT
from indic_eot.detector import ASSETS

SITE_MODEL = REPO_ROOT / "website" / "public" / "model"


def main() -> None:
    with open(ASSETS / "export.json", encoding="utf-8") as f:
        info = json.load(f)
    SITE_MODEL.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ASSETS / info["quantized_file"], SITE_MODEL / info["quantized_file"])
    filters = np.load(ASSETS / "mel_filters.npy")
    filters.astype("<f4").tofile(SITE_MODEL / "mel_filters.bin")
    summary = {
        "file": info["quantized_file"],
        "mel_filters": "mel_filters.bin",
        "mel_filters_shape": list(filters.shape),
        "max_seconds": info["max_seconds"],
        "sample_rate": info["sample_rate"],
    }
    (SITE_MODEL / "model.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

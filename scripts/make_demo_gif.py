"""Assemble demo frames (PNG) into a compact README GIF.

    python scripts/make_demo_gif.py <frames-dir> docs/images/demo.gif

Downscales, drops near-identical consecutive frames (the long "thinking" pause on a CPU),
and holds the last frame so viewers can read the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops


def main(frames_dir: str, out_path: str, width: int = 960, frame_ms: int = 220) -> None:
    paths = sorted(Path(frames_dir).glob("*.png"))
    frames: list[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        if frames and ImageChops.difference(img, frames[-1]).getbbox() is None:
            continue  # identical to the previous frame
        frames.append(img)
    palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    durations = [frame_ms] * len(palette)
    durations[-1] = 4000
    palette[0].save(out_path, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"{len(palette)} frames (from {len(paths)}) -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:3])

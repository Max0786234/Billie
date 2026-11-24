 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif"""Assemble demo frames (PNG) into a compact README GIF.
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    python scripts/make_demo_gif.py <frames-dir> docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gifDownscales, drops near-identical consecutive frames (the long "thinking" pause on a CPU),
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gifand holds the last frame so viewers can read the answer.
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif"""
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.giffrom __future__ import annotations
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gifimport sys
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.giffrom pathlib import Path
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.giffrom PIL import Image, ImageChops
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gifdef main(frames_dir: str, out_path: str, width: int = 960, frame_ms: int = 220) -> None:
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    paths = sorted(Path(frames_dir).glob("*.png"))
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    frames: list[Image.Image] = []
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    for path in paths:
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif        img = Image.open(path).convert("RGB")
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif        if frames and ImageChops.difference(img, frames[-1]).getbbox() is None:
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif            continue  # identical to the previous frame
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif        frames.append(img)
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    durations = [frame_ms] * len(palette)
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    durations[-1] = 4000
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    palette[0].save(out_path, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True)
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    size_mb = Path(out_path).stat().st_size / 1e6
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    print(f"{len(palette)} frames (from {len(paths)}) -> {out_path} ({size_mb:.1f} MB)")
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gifif __name__ == "__main__":
 Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif    main(*sys.argv[1:3])

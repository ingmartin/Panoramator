from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "public-demo"

FRAME_WIDTH = 320
FRAME_HEIGHT = 220
FPS = 12


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _gradient_background(width: int, height: int, start: tuple[int, int, int], end: tuple[int, int, int]) -> np.ndarray:
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        alpha = x / max(width - 1, 1)
        color = np.array(start) * (1.0 - alpha) + np.array(end) * alpha
        bg[:, x] = color.astype(np.uint8)
    return bg


def _draw_label(img: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.8) -> None:
    cv2.putText(
        img,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_linear_demo() -> None:
    canvas_w = 1200
    canvas_h = FRAME_HEIGHT
    canvas = _gradient_background(canvas_w, canvas_h, (28, 58, 96), (198, 111, 45))
    for idx in range(10):
        x0 = 40 + idx * 110
        color = (40 + idx * 18, 220 - idx * 12, 130 + idx * 8)
        cv2.rectangle(canvas, (x0, 36), (x0 + 72, 184), color, -1)
        cv2.circle(canvas, (x0 + 36, 110), 22, (255, 255, 255), 2)
        _draw_label(canvas, f"P{idx+1}", (x0 + 10, 210), 0.65)
    _draw_label(canvas, "LINEAR PANORAMA DEMO", (30, 30), 0.9)

    frames: list[np.ndarray] = []
    positions = np.linspace(0, canvas_w - FRAME_WIDTH, 42)
    for i, x in enumerate(positions):
        x0 = int(round(float(x)))
        frame = canvas[:, x0 : x0 + FRAME_WIDTH].copy()
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (255, 255, 255), 2)
        _draw_label(frame, f"build linear  {i+1:02d}", (12, 24), 0.55)
        frames.append(frame)

    cv2.imwrite(str(OUT_DIR / "build-linear-reference.png"), canvas)
    cv2.imwrite(str(OUT_DIR / "build-linear-preview.png"), frames[len(frames) // 2])
    _write_video(OUT_DIR / "build-linear-input.mp4", frames)


def make_rotation_demo() -> None:
    strip_w = 1500
    strip_h = FRAME_HEIGHT
    strip = _gradient_background(strip_w, strip_h, (20, 98, 82), (26, 32, 88))
    for idx in range(14):
        x = 20 + idx * 105
        cv2.ellipse(strip, (x + 40, 110), (42, 72), 0, 0, 360, (240, 240 - idx * 8, 80 + idx * 7), -1)
        _draw_label(strip, f"R{idx+1}", (x + 8, 115), 0.6)
    _draw_label(strip, "ROTATION DEMO", (24, 28), 0.9)

    frames: list[np.ndarray] = []
    positions = np.linspace(0, strip_w - FRAME_WIDTH, 48)
    for i, x in enumerate(positions):
        x0 = int(round(float(x)))
        frame = strip[:, x0 : x0 + FRAME_WIDTH].copy()
        fade = np.tile(np.linspace(0.72, 1.0, FRAME_WIDTH, dtype=np.float32), (FRAME_HEIGHT, 1))
        frame = np.clip(frame.astype(np.float32) * fade[:, :, None], 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (255, 255, 255), 2)
        _draw_label(frame, f"build rotation  {i+1:02d}", (12, 24), 0.55)
        frames.append(frame)

    cv2.imwrite(str(OUT_DIR / "build-rotation-reference.png"), strip)
    cv2.imwrite(str(OUT_DIR / "build-rotation-preview.png"), frames[len(frames) // 2])
    _write_video(OUT_DIR / "build-rotation-input.mp4", frames)


def _make_cylinder_texture(width: int, height: int) -> np.ndarray:
    texture = _gradient_background(width, height, (170, 62, 42), (237, 196, 54))
    for idx in range(8):
        x = 30 + idx * 70
        cv2.rectangle(texture, (x, 20), (x + 36, height - 20), (40, 40, 40), -1)
        cv2.circle(texture, (x + 18, height // 2), 11, (255, 255, 255), 2)
    _draw_label(texture, "CYLINDER LABEL TEXTURE", (20, 34), 0.8)
    return texture


def make_unwrap_demo() -> None:
    texture_w = 640
    texture_h = 180
    texture = _make_cylinder_texture(texture_w, texture_h)
    frames: list[np.ndarray] = []

    for i, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, 54, endpoint=False)):
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 245, dtype=np.uint8)
        cx = FRAME_WIDTH // 2
        cy = FRAME_HEIGHT // 2
        cylinder_w = 138
        cylinder_h = 164
        rx = cylinder_w / 2.0
        ry = cylinder_h / 2.0

        for x in range(FRAME_WIDTH):
            dx = (x - cx) / max(rx, 1.0)
            if abs(dx) > 1.0:
                continue
            theta = math.asin(dx)
            u = ((theta + angle) / (2.0 * math.pi)) % 1.0
            tex_x = min(texture_w - 1, max(0, int(round(u * (texture_w - 1)))))
            brightness = 0.35 + 0.65 * math.cos(theta) ** 1.5
            col = texture[:, tex_x].astype(np.float32) * brightness
            top = int(round(cy - ry))
            bottom = int(round(cy + ry))
            frame[top:bottom, x] = cv2.resize(col[:, None, :], (1, bottom - top), interpolation=cv2.INTER_LINEAR)[:, 0, :]

        cv2.ellipse(frame, (cx, cy), (int(rx), int(ry)), 0, 0, 360, (40, 40, 40), 2)
        cv2.ellipse(frame, (cx, cy - int(ry)), (int(rx), 10), 0, 0, 360, (90, 90, 90), 2)
        cv2.ellipse(frame, (cx, cy + int(ry)), (int(rx), 10), 0, 0, 360, (90, 90, 90), 2)
        _draw_label(frame, f"unwrap cylinder  {i+1:02d}", (12, 24), 0.55)
        frames.append(frame)

    cv2.imwrite(str(OUT_DIR / "unwrap-cylinder-reference.png"), texture)
    cv2.imwrite(str(OUT_DIR / "unwrap-cylinder-preview.png"), frames[len(frames) // 2])
    _write_video(OUT_DIR / "unwrap-cylinder-input.mp4", frames)


def make_overview() -> None:
    preview_names = [
        ("build-linear-preview.png", "build / linear"),
        ("build-rotation-preview.png", "build / rotation"),
        ("unwrap-cylinder-preview.png", "unwrap / cylinder"),
    ]
    tiles: list[np.ndarray] = []
    for name, label in preview_names:
        image = cv2.imread(str(OUT_DIR / name))
        if image is None:
            raise RuntimeError(f"missing preview image: {name}")
        tile = cv2.copyMakeBorder(image, 0, 42, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        _draw_label(tile, label, (14, tile.shape[0] - 14), 0.7)
        tiles.append(tile)

    gap = np.full((tiles[0].shape[0], 18, 3), 255, dtype=np.uint8)
    overview = np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], axis=1)
    cv2.imwrite(str(OUT_DIR / "overview.png"), overview)


def main() -> None:
    _ensure_dir(OUT_DIR)
    make_linear_demo()
    make_rotation_demo()
    make_unwrap_demo()
    make_overview()


if __name__ == "__main__":
    main()

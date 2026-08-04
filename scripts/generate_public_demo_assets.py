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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


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


def _apply_vignette(frame: np.ndarray, strength: float = 0.2) -> np.ndarray:
    yy, xx = np.indices((frame.shape[0], frame.shape[1]), dtype=np.float32)
    cx = (frame.shape[1] - 1) / 2.0
    cy = (frame.shape[0] - 1) / 2.0
    dx = (xx - cx) / max(cx, 1.0)
    dy = (yy - cy) / max(cy, 1.0)
    radius = np.sqrt(dx * dx + dy * dy)
    vignette = 1.0 - strength * np.clip(radius, 0.0, 1.0) ** 1.7
    return np.clip(frame.astype(np.float32) * vignette[:, :, None], 0, 255).astype(np.uint8)


def _apply_edge_warp(frame: np.ndarray, phase: float) -> np.ndarray:
    height, width = frame.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    x_norm = (xx / max(width - 1, 1)) * 2.0 - 1.0
    y_norm = (yy / max(height - 1, 1)) * 2.0 - 1.0
    bow = 12.0 * np.sin(phase * 2.0 * math.pi) * (x_norm**2) * (1.0 - 0.25 * y_norm**2)
    squeeze = 8.0 * np.cos(phase * 2.0 * math.pi) * y_norm * np.abs(x_norm)
    map_x = (xx + bow).astype(np.float32)
    map_y = (yy + squeeze).astype(np.float32)
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _apply_scanlines(frame: np.ndarray, strength: float = 0.12) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    rows = np.arange(frame.shape[0], dtype=np.float32)
    band = 1.0 - strength * (0.5 + 0.5 * np.sin(rows * 0.42))
    return np.clip(frame_f * band[:, None, None], 0, 255).astype(np.uint8)


def _stylize_rotation_reference(strip: np.ndarray) -> np.ndarray:
    height, width = strip.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    x_norm = (xx / max(width - 1, 1)) * 2.0 - 1.0
    y_norm = (yy / max(height - 1, 1)) * 2.0 - 1.0
    wave_y = 9.0 * np.sin((x_norm + 1.0) * math.pi * 2.6) * (0.35 + 0.65 * (1.0 - y_norm**2))
    wave_x = 10.0 * np.sin(y_norm * math.pi * 0.9) * (x_norm**2)
    map_x = (xx + wave_x).astype(np.float32)
    map_y = (yy + wave_y).astype(np.float32)
    warped = cv2.remap(
        strip,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    light = 0.88 + 0.18 * np.cos((x_norm + 0.15) * math.pi)
    warped = np.clip(warped.astype(np.float32) * light[:, :, None], 0, 255).astype(np.uint8)
    warped = _apply_scanlines(warped, 0.08)
    cv2.putText(
        warped,
        "synthetic cylindrical result",
        (width - 420, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return warped


def _render_rotation_frame(strip: np.ndarray, phase: float, index: int) -> np.ndarray:
    strip_w = strip.shape[1]
    eased = 0.5 - 0.5 * math.cos(math.pi * phase)
    center = eased * (strip_w - FRAME_WIDTH)
    zoom = 1.0 + 0.22 * math.sin(phase * 2.0 * math.pi) ** 2
    crop_w = max(FRAME_WIDTH, min(strip_w, int(np.rint(FRAME_WIDTH / zoom))))
    x0 = int(np.clip(np.rint(center), 0, strip_w - crop_w))
    view = strip[:, x0 : x0 + crop_w].copy()
    frame = cv2.resize(view, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
    frame = _apply_edge_warp(frame, phase)
    frame = _apply_vignette(frame, 0.34)
    frame = _apply_scanlines(frame, 0.1)

    fade = np.tile(np.linspace(0.72, 1.0, FRAME_WIDTH, dtype=np.float32), (FRAME_HEIGHT, 1))
    exposure = 0.9 + 0.16 * math.sin(phase * 2.0 * math.pi - 0.5)
    frame = np.clip(frame.astype(np.float32) * fade[:, :, None] * exposure, 0, 255).astype(np.uint8)

    blur_kernel = 3 + 2 * int((math.sin(phase * 2.0 * math.pi) + 1.0) > 1.6)
    if blur_kernel > 3:
        frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0.0)

    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (255, 255, 255), 2)
    _draw_label(frame, f"build rotation  {index+1:02d}", (12, 24), 0.55)
    _draw_label(frame, f"yaw {(-42.0 + 84.0 * eased):+05.1f} deg", (12, 204), 0.5)
    return frame


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
    for i, position in enumerate(positions):
        x0 = int(np.rint(position))
        frame = canvas[:, x0 : x0 + FRAME_WIDTH].copy()
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (255, 255, 255), 2)
        _draw_label(frame, f"build linear  {i+1:02d}", (12, 24), 0.55)
        frames.append(frame)

    _write_image(OUT_DIR / "build-linear-reference.png", canvas)
    _write_image(OUT_DIR / "build-linear-preview.png", frames[len(frames) // 2])
    _write_video(OUT_DIR / "build-linear-input.mp4", frames)


def make_rotation_demo() -> None:
    strip_w = 1500
    strip_h = FRAME_HEIGHT
    strip = _gradient_background(strip_w, strip_h, (20, 98, 82), (26, 32, 88))
    for idx in range(14):
        x = 20 + idx * 105
        cv2.ellipse(strip, (x + 40, 110), (42, 72), 0, 0, 360, (240, 240 - idx * 8, 80 + idx * 7), -1)
        _draw_label(strip, f"R{idx+1}", (x + 8, 115), 0.6)
        top = 22 + (idx % 3) * 8
        cv2.line(strip, (x + 16, top), (x + 64, top), (255, 255, 255), 2)
        cv2.line(strip, (x + 40, 28), (x + 40, 190), (18, 18, 18), 1)
    for idx in range(7):
        x = 70 + idx * 210
        cv2.rectangle(strip, (x, 44), (x + 26, 176), (245, 245, 245), 2)
        cv2.circle(strip, (x + 13, 84), 8, (70, 200, 255), -1)
    _draw_label(strip, "ROTATION DEMO", (24, 28), 0.9)

    frames: list[np.ndarray] = []
    phases = np.linspace(0.0, 1.0, 48)
    for i, phase in enumerate(phases):
        frames.append(_render_rotation_frame(strip, float(phase), i))

    _write_image(OUT_DIR / "build-rotation-reference.png", _stylize_rotation_reference(strip))
    _write_image(OUT_DIR / "build-rotation-preview.png", frames[len(frames) // 2])
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
            tex_x = min(texture_w - 1, max(0, int(np.rint(u * (texture_w - 1)))))
            brightness = 0.35 + 0.65 * math.cos(theta) ** 1.5
            col = texture[:, tex_x].astype(np.float32) * brightness
            top = int(np.rint(cy - ry))
            bottom = int(np.rint(cy + ry))
            frame[top:bottom, x] = cv2.resize(col[:, None, :], (1, bottom - top), interpolation=cv2.INTER_LINEAR)[:, 0, :]

        cv2.ellipse(frame, (cx, cy), (int(rx), int(ry)), 0, 0, 360, (40, 40, 40), 2)
        cv2.ellipse(frame, (cx, cy - int(ry)), (int(rx), 10), 0, 0, 360, (90, 90, 90), 2)
        cv2.ellipse(frame, (cx, cy + int(ry)), (int(rx), 10), 0, 0, 360, (90, 90, 90), 2)
        _draw_label(frame, f"unwrap cylinder  {i+1:02d}", (12, 24), 0.55)
        frames.append(frame)

    _write_image(OUT_DIR / "unwrap-cylinder-reference.png", texture)
    _write_image(OUT_DIR / "unwrap-cylinder-preview.png", frames[len(frames) // 2])
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
    _write_image(OUT_DIR / "overview.png", overview)


def main() -> None:
    _ensure_dir(OUT_DIR)
    make_linear_demo()
    make_rotation_demo()
    make_unwrap_demo()
    make_overview()


if __name__ == "__main__":
    main()

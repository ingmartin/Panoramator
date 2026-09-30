# Changelog

## 0.5.1

- Hardened the unwrap pipeline with a shared `SurfaceBuild` model and clearer
  builder boundaries.
- Ensured video sources are closed when opening or metadata reading fails.
- Rejected malformed frame inputs with diagnostics instead of aborting the
  entire batch.
- Narrowed recoverable surface-builder fallbacks and preserved unexpected
  programming errors for visibility.
- Improved NumPy/OpenCV typing and added regression coverage for these failure
  paths.

## 0.5.0

- Added `FrameSource`, `DirectoryFrameSource`, and BGR `FrameInput` support.
- Added `PanoramaBuilder.build_from_frames()` for lazy frame sequences without
  an intermediate MP4.
- Added progress, cancellation, canvas-limit diagnostics, and atomic output
  replacement.
- Preserved the existing `build_from_video()` and CLI contracts.

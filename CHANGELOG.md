# Changelog

## 0.5.0

- Added `FrameSource`, `DirectoryFrameSource`, and BGR `FrameInput` support.
- Added `PanoramaBuilder.build_from_frames()` for lazy frame sequences without
  an intermediate MP4.
- Added progress, cancellation, canvas-limit diagnostics, and atomic output
  replacement.
- Preserved the existing `build_from_video()` and CLI contracts.

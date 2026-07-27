import numpy as np
import pytest

from panoramator.canvas.builder import PanoramaCanvasBuilder
from panoramator.config.models import PanoramaConfig


def test_canvas_builder_rejects_excessive_canvas_size() -> None:
    config = PanoramaConfig(max_canvas_width=100, max_canvas_height=100)
    builder = PanoramaCanvasBuilder(config)
    frame_shapes = [(50, 50), (50, 50)]
    homographies = [
        np.eye(3, dtype=np.float64),
        np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    ]

    with pytest.raises(RuntimeError, match="Canvas size"):
        builder.build(frame_shapes, homographies)


def test_canvas_builder_offsets_negative_coordinates() -> None:
    builder = PanoramaCanvasBuilder(PanoramaConfig())
    homography = np.array([[1.0, 0.0, -4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    canvas = builder.build([(10, 20)], [homography])

    assert (canvas.width, canvas.height) == (20, 10)
    assert np.allclose(canvas.offset_matrix, np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]))
    assert canvas.global_homographies == [homography]

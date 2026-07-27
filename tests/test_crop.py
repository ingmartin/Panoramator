import numpy as np

from panoramator.postprocess.crop import crop_black_borders, crop_to_visible_area


def test_crop_black_borders_reduces_empty_border() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 4:18] = 255
    cropped = crop_black_borders(image)
    assert cropped.shape[:2] == (10, 14)


def test_crop_to_visible_area_removes_internal_black_corners() -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[1:11, 1:11] = 255
    image[1:4, 1:4] = 0
    image[8:11, 8:11] = 0

    regular_crop = crop_black_borders(image)
    photo_crop = crop_to_visible_area(image)

    assert regular_crop.shape[:2] == (10, 10)
    assert photo_crop.shape[:2] == (7, 7)
    assert np.all(photo_crop > 0)


def test_crop_to_visible_area_uses_mask_instead_of_black_pixels() -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[1:11, 1:11] = 180
    image[4:8, 4:8] = 0
    visible_mask = np.zeros((12, 12), dtype=np.uint8)
    visible_mask[1:11, 1:11] = 255

    photo_crop = crop_to_visible_area(image, visible_mask)

    assert photo_crop.shape[:2] == (10, 10)
    assert np.any(photo_crop == 0)

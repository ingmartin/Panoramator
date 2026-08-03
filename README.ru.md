<p align="center">
  <img src="assets/logo.svg" width="180" alt="Panoramator">
</p>

<h1 align="center">Panoramator</h1>

<p align="center">
Python-пакет для построения панорам из видео с расширяемой архитектурой.
</p>

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)
![Code style](https://img.shields.io/badge/code%20style-ruff-purple.svg)
![PyPI](https://img.shields.io/pypi/v/panoramator.svg)

</div>


## Быстрый запуск

### Установка

```bash
pip install panoramator
```

### Построить панораму

```bash
panoramator build video.mp4 output.png
```

### Какую команду использовать

* `build` - для панорам сцены при примерно линейном движении камеры или повороте на месте.
* `unwrap` - для облёта одного объекта, когда нужен не снимок сцены, а развёртка наблюдаемой поверхности.

### Другие примеры

```bash
panoramator build VID_20260709_140742.mp4 output.png
```

Для развёртки наблюдаемой поверхности используйте отдельную команду `unwrap`. Рядом с итоговым PNG она создаёт debug-директорию `*_debug` с `run.json`, `effective_config.json`, coverage, source/error map и промежуточными mosaic-артефактами:

```bash
panoramator unwrap VID20260729124935.mp4 surface.png --surface auto --allow-partial
```

Если нужна более строгая подрезка только по реально видимой полосе, включите `photo-mode` и для `unwrap`:

```bash
panoramator unwrap VID20260729124935.mp4 surface.png --surface auto --allow-partial --photo-mode --photo-crop-margin-px 5
```

`unwrap` принимает JSON-конфиг и точечные CLI-переопределения вроде `--sampling-step`, `--max-frames`, `--blur-threshold`, `--max-mosaic-boundary-mean-error` и других параметров rectification/gate, по той же схеме, что и `build`.

Чтобы не сохранять debug-артефакты ни для `build`, ни для `unwrap`, используйте `--no-save-debug-artifacts`.

Если фиксированный порог резкости слишком строгий:

```bash
panoramator build VID_20260709_140742.mp4 output.png --adaptive-blur-threshold
```

Если кадры лишь немного не дотягивают по резкости, лучше оставить adaptive threshold и подстроить rescue sharpening, а не слишком сильно снижать blur threshold:

```bash
panoramator build VID_20260709_140742.mp4 output.png --adaptive-blur-threshold --blur-rescue-sharpen-strength 0.25 --blur-rescue-sharpen-sigma 1.0
```

Если нужен компромисс в пользу качества при разумной скорости, уменьшайте только разрешение для признаков и включайте оконный выбор кадров:

```bash
panoramator build VID_20260709_140742.mp4 output.png --feature-downscale 0.5 --frame-selection-window-size 3
```

Если на панораме заметны швы, можно отдельно подстроить feather width и очень локальное seam blur:

```bash
panoramator build VID_20260709_140742.mp4 output.png --seam-blur-kernel 7 --seam-band-width 9 --feather-blend-kernel 25
```

Для поворота камеры на месте используйте криволинейную поверхность и при необходимости задайте калибровку камеры:

```bash
panoramator build input.mp4 output.png --capture-mode rotation --horizontal-fov-degrees 70
```

Если нужен кадр без чёрных углов, включите `photo-mode`. Для криволинейных панорам он специально подрезает небольшой край маски, и этот отступ можно увеличить:

```bash
panoramator build input.mp4 output.png --capture-mode rotation --photo-mode --photo-crop-margin-px 5
```

Режим съёмки и проекция задаются независимо: `--capture-mode` принимает `auto`, `linear`
или `rotation`, а `--projection` — `auto`, `planar`, `cylindrical`, `spherical`.
Неоднозначный автоматический выбор сохраняет совместимый pipeline `linear + planar`.
Для поворота камеры на месте задайте `--capture-mode rotation`: он выбирает цилиндрическую
поверхность. Для облёта одного объекта используйте `unwrap`, а не `build`. Модель камеры
можно уточнить через `--focal-length-px` или `--horizontal-fov-degrees`.

## Статус

Пакет опубликован на PyPI и доступен для установки через pip.

## Лицензия

Репозиторий публикуется под лицензией MIT. См. файл `LICENSE`.

## Возможности текущей версии

* чтение видео через OpenCV;
* построение панорам сцены для `linear` и `rotation`;
* pipeline `unwrap` для развёртки наблюдаемой поверхности при облёте объекта;
* извлечение признаков ORB или SIFT с валидацией геометрии и fallback по sampling;
* плоская, цилиндрическая и сферическая проекции с необязательной калибровкой камеры;
* photometric normalization, обработка швов, политики crop и sharpening;
* debug-артефакты и локальные private acceptance-проверки.

## Практические замечания

По умолчанию используется модель движения `affine`. Для многих видеопанорам стоит отдельно попробовать `partial_affine`, если нужно сильнее ограничить деформации между кадрами.

По умолчанию `blur_threshold` фиксированный. Опция `--adaptive-blur-threshold` включает адаптивный режим, в котором фактический порог понижается по распределению резкости sampled-кадров текущего видео.

## Установка

```bash
python -m pip install panoramator
```

## Использование как Python-пакета

Модуль можно подключать к другому Python-приложению как обычный пакет. После установки в окружение достаточно импортировать основные классы и запустить построение панорамы:

```python
from pathlib import Path

from panoramator import PanoramaBuilder, PanoramaConfig

config = PanoramaConfig(
    sampling_step=15,
    max_frames=25,
    motion_model="partial_affine",
    crop_result=True,
)

builder = PanoramaBuilder(config)
result = builder.build_from_video(
    video_path=Path("VID_20260709_140742.mp4"),
    output_path=Path("output.png"),
)

print(result.metadata)
print(result.diagnostics.status)
print(result.diagnostics.output_files)
```

## Установка для разработки

```bash
python -m pip install -e ".[dev]"
```

## Тесты

```bash
pytest -q
```

Private acceptance-фикстуры намеренно отделены от обычного воспроизводимого набора. Для локальной проверки конфиденциальных видео:

```bash
PANORAMATOR_RUN_PRIVATE_VIDEO=1 python3 -m pytest tests/test_private_orbit_acceptance.py -q
PANORAMATOR_RUN_PRIVATE_VIDEO=1 python3 -m pytest tests/test_private_panorama_acceptance.py -q
```

## Параметры конфигурации

Все параметры задаются через `PanoramaConfig`, JSON-конфиг или частично через CLI.

### Входные кадры

* `sampling_step` - шаг по индексам кадров при первичной выборке из видео. Меньше значение: больше кадров и выше шанс найти удачную пару, но дольше обработка. По умолчанию `15`.
* `max_frames` - максимальное число sampled-кадров, которые попадут в дальнейший pipeline. По умолчанию `25`.
* `downscale` - масштаб уменьшения кадров перед обработкой и финальным warp. `1.0` означает исходный размер. По умолчанию `1.0`.
* `feature_downscale` - дополнительный масштаб только для feature extraction и matching. Позволяет оставить итоговую панораму в полном разрешении, но удешевить геометрию. По умолчанию `1.0`.

### Отбор кадров

* `blur_threshold` - фиксированный порог резкости. Кадры с меньшей оценкой считаются слишком размытыми. По умолчанию `80.0`.
* `adaptive_blur_threshold` - включает адаптивное смягчение `blur_threshold` по распределению резкости в конкретном видео. По умолчанию `False`.
* `adaptive_blur_percentile` - квантиль, по которому вычисляется адаптивный порог резкости. Используется только если включён `adaptive_blur_threshold`. По умолчанию `0.35`.
* `enable_blur_rescue_sharpening` - перед отбрасыванием мягкого кадра пробует слегка поднять резкость через unsharp mask. По умолчанию `True`.
* `blur_rescue_sharpen_strength` - сила rescue-sharpening. Больше значение: агрессивнее усиление. По умолчанию `0.2`.
* `blur_rescue_sharpen_sigma` - sigma гауссова размытия внутри rescue-sharpening. По умолчанию `1.0`.
* `frame_selection_window_size` - выбирает самый резкий валидный кадр внутри локального окна вместо того, чтобы брать каждый подходящий кадр подряд. По умолчанию `1`.
* `min_difference` - минимальное отличие нового кадра от предыдущего выбранного. Помогает не брать почти одинаковые кадры. По умолчанию `8.0`.

### Признаки и fallback

* `feature_backend` - основной backend признаков. Сейчас поддерживаются `orb` и `sift`. По умолчанию `orb`.
* `enable_feature_fallback` - включает автоматический fallback `ORB -> SIFT`, если получившаяся валидная цепочка слишком короткая. По умолчанию `True`.
* `fallback_feature_backend` - backend, который будет использован при feature fallback. По умолчанию `sift`.
* `fallback_min_chain_length` - минимальная длина валидной цепочки, ниже которой запускается feature fallback. По умолчанию `8`.
* `max_features` - максимальное количество ключевых точек, которые пытается извлечь detector. По умолчанию `2500`.
* `ratio_test` - коэффициент Lowe ratio test при фильтрации match-ей. Меньше значение: строже фильтрация. По умолчанию `0.75`.
* `min_match_count` - минимальное количество good matches для попытки оценить геометрию. По умолчанию `20`.
* `min_inlier_count` - минимальное число согласованных с RANSAC совпадений для принятия пары. По умолчанию `8`.
* `min_inlier_ratio` - минимальная доля good matches, которая должна быть inlier-ами RANSAC. По умолчанию `0.4`.

### Sampling fallback

* `enable_sampling_fallback` - разрешает повторный проход с более плотным шагом кадров, если основной проход дал слишком слабую цепочку. По умолчанию `True`.
* `fallback_sampling_step` - альтернативный, более плотный шаг кадров для fallback-попытки. Имеет смысл задавать меньше основного `sampling_step`. По умолчанию `8`.

### Геометрия

* `motion_model` - модель движения между кадрами. Поддерживаются `translation`, `partial_affine`, `affine`, `homography`. По умолчанию `affine`.
* `ransac_threshold` - порог reprojection error внутри RANSAC при оценке преобразования. По умолчанию `4.0`.
* `max_reprojection_error` - максимальная средняя reprojection error для принятия пары как валидной. По умолчанию `6.0`.
* `max_scale_deviation` - максимально допустимое отклонение масштаба от `1.0` в найденной трансформации. Защищает от неадекватных match-ей. По умолчанию `0.15`.
* `max_rotation_degrees` - максимально допустимый поворот между соседними кадрами. По умолчанию `12.0`.
* `max_homography_corner_scale` - максимально допустимый размер проекции одного кадра относительно исходного в режиме `homography`. По умолчанию `2.0`.

### Режим съёмки, проекция и камера

* `capture_mode` - `auto`, `linear` или `rotation`. Для облёта одного объекта используйте `unwrap`, а не `build`. `auto` намеренно консервативен и при неоднозначности выбирает `linear`. По умолчанию `auto`.
* `projection` - `auto`, `planar`, `cylindrical` или `spherical`. Явно заданная проекция имеет приоритет над автоматическим выбором. По умолчанию `auto`.
* `focal_length_px` или `horizontal_fov_degrees` - взаимоисключающие необязательные параметры калибровки камеры для криволинейных проекций.
* `projection_center_x`, `projection_center_y` - необязательная главная точка в пикселях исходного кадра.
* `projection_contour_samples` - число точек контура для расчёта криволинейного холста. По умолчанию `32`.

### Холст и stitching

* `max_canvas_width` - жёсткий лимит ширины итогового холста. Защищает от раздувания памяти. По умолчанию `12000`.
* `max_canvas_height` - жёсткий лимит высоты итогового холста. По умолчанию `12000`.

### Blending и швы

* `feather_blend_kernel` - ширина сглаживания весовой маски около границ warped-кадров. По умолчанию `21`.
* `seam_blur_kernel` - сила локального blur вдоль seam-зоны. По умолчанию `1` (выключен), чтобы не замыливать детали на стыке кадров.
* `seam_band_width` - ширина полосы вокруг линии шва, где разрешено локальное сглаживание. По умолчанию `7`.
* `enable_photometric_normalization` - подравнивает яркость и контраст соседних выбранных кадров до warp. По умолчанию `True`.
* `photometric_smoothing` - насколько сильно соседние кадры тянутся друг к другу по яркости/контрасту. По умолчанию `0.65`.
* `enable_global_photometric_normalization` — явное включение якорной коррекции overlap по всей curved-цепочке. По умолчанию `False`; CLI: `--global-photometric-normalization`.
* `overlap_sharpness_weight` - насколько blending должен предпочитать более детальные участки в overlap-зонах. По умолчанию `0.35`.
* `rotation_min_baseline_px` - минимальное накопленное смещение для сохранения следующего keyframe в `rotation`. По умолчанию `12.0`.
* `rotation_min_new_coverage_ratio` - минимальная доля новой маски перед созданием нового seam в криволинейной панораме. По умолчанию `0.01`.
* `photometric_gain_limit`, `photometric_offset_limit` - защитные пределы цветовой коррекции curved-overlap. По умолчанию `0.12`, `20.0`.

### Постобработка и артефакты

* `crop_result` - включает автообрезку чёрных полей после stitching. По умолчанию `True`.
* `photo_mode` - строго обрезает до максимального прямоугольника внутри видимой области для любой проекции. В криволинейной панораме маска дополнительно эродируется на `photo_crop_margin_px`, поэтому чистые края достигаются ценой части полезной площади. По умолчанию `False`.
* `crop_policy` - `auto`, `bounding`, `inscribed_rectangle` или `preserve_alpha`. Явная политика имеет приоритет над автоматической. По умолчанию `auto`.
* `max_inscribed_crop_loss`, `max_inscribed_crop_width_loss` - пороги безопасности для явного строгого crop вне `photo_mode`. По умолчанию `0.35`, `0.25`.
* `photo_crop_margin_px` - внутренний отступ кропа в curved `photo_mode`. По умолчанию `3`.
* `enable_narrow_gap_fill`, `max_narrow_gap_width` - заполняют только замкнутые горизонтальные разрывы маски до указанной ширины в криволинейной панораме. По умолчанию `True`, `4`.
* `enable_final_sharpening` - включает мягкий финальный unsharp mask для уже собранной панорамы. По умолчанию `True`.
* `final_sharpen_strength` - сила финального sharpening. По умолчанию `0.15`.
* `final_sharpen_sigma` - sigma размытия внутри финального sharpening. По умолчанию `1.0`.
* `save_debug_artifacts` - сохраняет debug-директорию с effective config и отчётом запуска. Работает и для `build`, и для `unwrap`. По умолчанию `True`.

### Пример полного конфига

```json
{
  "sampling_step": 15,
  "max_frames": 25,
  "downscale": 1.0,
  "feature_downscale": 1.0,
  "blur_threshold": 80.0,
  "adaptive_blur_threshold": false,
  "adaptive_blur_percentile": 0.35,
  "enable_blur_rescue_sharpening": true,
  "blur_rescue_sharpen_strength": 0.2,
  "blur_rescue_sharpen_sigma": 1.0,
  "frame_selection_window_size": 1,
  "min_difference": 8.0,
  "feature_backend": "orb",
  "enable_feature_fallback": true,
  "fallback_feature_backend": "sift",
  "fallback_min_chain_length": 8,
  "enable_sampling_fallback": true,
  "fallback_sampling_step": 8,
  "max_features": 2500,
  "ratio_test": 0.75,
  "min_match_count": 20,
  "min_inlier_count": 8,
  "min_inlier_ratio": 0.4,
  "motion_model": "affine",
  "capture_mode": "auto",
  "projection": "auto",
  "focal_length_px": null,
  "horizontal_fov_degrees": null,
  "projection_center_x": null,
  "projection_center_y": null,
  "projection_contour_samples": 32,
  "ransac_threshold": 4.0,
  "max_reprojection_error": 6.0,
  "max_scale_deviation": 0.15,
  "max_rotation_degrees": 12.0,
  "max_homography_corner_scale": 2.0,
  "max_canvas_width": 12000,
  "max_canvas_height": 12000,
  "feather_blend_kernel": 21,
  "seam_blur_kernel": 1,
  "seam_band_width": 7,
  "enable_photometric_normalization": true,
  "enable_global_photometric_normalization": false,
  "photometric_smoothing": 0.65,
  "overlap_sharpness_weight": 0.35,
  "rotation_min_baseline_px": 12.0,
  "rotation_min_new_coverage_ratio": 0.01,
  "photometric_gain_limit": 0.12,
  "photometric_offset_limit": 20.0,
  "enable_narrow_gap_fill": true,
  "max_narrow_gap_width": 4,
  "photo_crop_margin_px": 3,
  "crop_result": true,
  "photo_mode": false,
  "crop_policy": "auto",
  "max_inscribed_crop_loss": 0.35,
  "max_inscribed_crop_width_loss": 0.25,
  "trajectory_smoothing_window": 5,
  "max_rotation_scale_correction": 0.02,
  "enable_final_sharpening": true,
  "final_sharpen_strength": 0.15,
  "final_sharpen_sigma": 1.0,
  "save_debug_artifacts": true
}
```


`unwrap` теперь принимает JSON-конфиг и точечные overrides в том же стиле, что и `build`: например `--sampling-step`, `--max-frames`, `--blur-threshold`, `--max-mosaic-boundary-mean-error` и параметры quality gate / rectification.

Чтобы отключить debug-артефакты для любой из команд, используйте `--no-save-debug-artifacts`.

Для видео, где фиксированный порог резкости слишком строгий:

```bash
panoramator build VID_20260709_140742.mp4 output.png --adaptive-blur-threshold
```

Если кадры лишь немного «мыльные», лучше не опускать порог слишком сильно, а слегка настроить rescue-sharpening:

```bash
panoramator build VID_20260709_140742.mp4 output.png --adaptive-blur-threshold --blur-rescue-sharpen-strength 0.25 --blur-rescue-sharpen-sigma 1.0
```

Если нужен практичный компромисс между скоростью и качеством, можно уменьшить только разрешение для признаков и включить оконный выбор самого резкого кадра:

```bash
panoramator build VID_20260709_140742.mp4 output.png --feature-downscale 0.5 --frame-selection-window-size 3
```

Если на панораме заметны линии склейки, можно отдельно управлять шириной feather-зоны и очень локальным blur вдоль seam:

```bash
panoramator build VID_20260709_140742.mp4 output.png --seam-blur-kernel 7 --seam-band-width 9 --feather-blend-kernel 25
```

Для поворота камеры на месте используйте криволинейную поверхность и при возможности укажите калибровку камеры:

```bash
panoramator build input.mp4 output.png --capture-mode rotation --horizontal-fov-degrees 70
```

Если нужен кадр без чёрных углов, включите `photo-mode`. Для криволинейной панорамы он намеренно отрежет небольшой край; увеличьте отступ, если артефакты границы остаются:

```bash
panoramator build input.mp4 output.png --capture-mode rotation --photo-mode --photo-crop-margin-px 5
```

# panoramator

Python-пакет для построения панорам из видео с расширяемой архитектурой.

## Статус

Пакет опубликован на PyPI и доступен для установки через pip.

## Лицензия

Репозиторий публикуется под лицензией MIT. См. файл `LICENSE`.

## Возможности текущей версии

* чтение видео через OpenCV;
* отбор ключевых кадров по шагу, резкости и простому отличию;
* лёгкое условное повышение резкости кадра, если он лишь немного не проходит порог sharpness;
* опциональное извлечение признаков на уменьшенной копии кадра при full-resolution warp/blend;
* более умный локальный отбор кадров по самому резкому кандидату в окне;
* извлечение признаков ORB или SIFT;
* сопоставление признаков и оценка геометрии с `translation` / `partial_affine` / `affine` / `homography`;
* построение общего холста;
* feather-blending с дополнительным сглаживанием seam-зон, photometric normalization и учётом локальной резкости в overlap-зонах;
* автообрезка чёрных полей;
* опциональный `photo-mode` для кропа до полностью видимого прямоугольника без внутренних чёрных клиньев;
* мягкий финальный sharpening уже готовой панорамы;
* fallback `ORB -> SIFT`, если цепочка валидных кадров получилась слишком короткой;
* fallback на более плотный `sampling_step`, если нужно подобрать более резкие кадры;
* CLI для запуска и диагностики.

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
```

## Установка для разработки

```bash
python -m pip install -e ".[dev]"
```

## Тесты

```bash
pytest -q
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

### Холст и stitching

* `max_canvas_width` - жёсткий лимит ширины итогового холста. Защищает от раздувания памяти. По умолчанию `12000`.
* `max_canvas_height` - жёсткий лимит высоты итогового холста. По умолчанию `12000`.

### Blending и швы

* `feather_blend_kernel` - ширина сглаживания весовой маски около границ warped-кадров. По умолчанию `21`.
* `seam_blur_kernel` - сила локального blur вдоль seam-зоны. По умолчанию `5`.
* `seam_band_width` - ширина полосы вокруг линии шва, где разрешено локальное сглаживание. По умолчанию `7`.
* `enable_photometric_normalization` - подравнивает яркость и контраст соседних выбранных кадров до warp. По умолчанию `True`.
* `photometric_smoothing` - насколько сильно соседние кадры тянутся друг к другу по яркости/контрасту. По умолчанию `0.65`.
* `overlap_sharpness_weight` - насколько blending должен предпочитать более детальные участки в overlap-зонах. По умолчанию `0.35`.

### Постобработка и артефакты

* `crop_result` - включает автообрезку чёрных полей после stitching. По умолчанию `True`.
* `photo_mode` - включает более агрессивный crop до максимального прямоугольника, целиком лежащего в видимой области панорамы. Полезно, когда нужно убрать все чёрные углы и клинья. По умолчанию `False`.
* `enable_final_sharpening` - включает мягкий финальный unsharp mask для уже собранной панорамы. По умолчанию `True`.
* `final_sharpen_strength` - сила финального sharpening. По умолчанию `0.15`.
* `final_sharpen_sigma` - sigma размытия внутри финального sharpening. По умолчанию `1.0`.
* `save_debug_artifacts` - сохраняет debug-директорию с effective config и отчётом запуска. По умолчанию `True`.

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
  "ransac_threshold": 4.0,
  "max_reprojection_error": 6.0,
  "max_scale_deviation": 0.15,
  "max_rotation_degrees": 12.0,
  "max_homography_corner_scale": 2.0,
  "max_canvas_width": 12000,
  "max_canvas_height": 12000,
  "feather_blend_kernel": 21,
  "seam_blur_kernel": 5,
  "seam_band_width": 7,
  "enable_photometric_normalization": true,
  "photometric_smoothing": 0.65,
  "overlap_sharpness_weight": 0.35,
  "crop_result": true,
  "photo_mode": false,
  "enable_final_sharpening": true,
  "final_sharpen_strength": 0.15,
  "final_sharpen_sigma": 1.0,
  "save_debug_artifacts": true
}
```

## Быстрый запуск

```bash
panoramator build VID_20260709_140742.mp4 output.png
```

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

Если нужен итоговый кадр без любых чёрных углов после warp, включите `photo-mode`:

```bash
panoramator build VID_20260709_140742.mp4 output.png --photo-mode
```

# Публичный Showcase

На этой странице собраны публичные, не содержащие конфиденциальных данных demo-артефакты, которые быстро объясняют два основных сценария Panoramator.

## Публичный Demo Pack

Файлы ниже синтетические и безопасны для публикации. Их можно использовать в README, GitHub preview, обсуждениях в issue и release notes.

| Сценарий | Входное видео | Preview-кадр | Эталонный результат |
| --- | --- | --- | --- |
| `build` при примерно линейном движении | [build-linear-input.mp4](public-demo/build-linear-input.mp4) | [build-linear-preview.png](public-demo/build-linear-preview.png) | [build-linear-reference.png](public-demo/build-linear-reference.png) |
| `build` при повороте камеры на месте | [build-rotation-input.mp4](public-demo/build-rotation-input.mp4) | [build-rotation-preview.png](public-demo/build-rotation-preview.png) | [build-rotation-reference.png](public-demo/build-rotation-reference.png) |
| `unwrap` для облёта одного объекта цилиндрического типа | [unwrap-cylinder-input.mp4](public-demo/unwrap-cylinder-input.mp4) | [unwrap-cylinder-preview.png](public-demo/unwrap-cylinder-preview.png) | [unwrap-cylinder-reference.png](public-demo/unwrap-cylinder-reference.png) |

Обзорное изображение: [overview.png](public-demo/overview.png)

## Зачем Эти Артефакты Нужны

Они решают три задачи для публичного репозитория:

* README может показывать конкретные входы и ожидаемые выходы без использования конфиденциальных видео.
* Участники проекта могут обсуждать поддерживаемые сценарии на общих, воспроизводимых примерах.
* Для релизов и публикаций можно ссылаться на реальные артефакты, а не только на абстрактные CLI-команды.

## Как Перегенерировать

```bash
python3 scripts/generate_public_demo_assets.py
```

Генератор достаточно детерминирован для обновления документации и специально держит размер файлов небольшим.

# Showcase

This page collects public, non-confidential demo assets that explain the two main Panoramator workflows quickly.

## Public Demo Pack

The files below are synthetic and safe to publish. They are meant for README links, GitHub previews, issue discussions, and release notes.

| Workflow | Input video | Preview frame | Reference output |
| --- | --- | --- | --- |
| `build` with approximately linear motion | [build-linear-input.mp4](public-demo/build-linear-input.mp4) | [build-linear-preview.png](public-demo/build-linear-preview.png) | [build-linear-reference.png](public-demo/build-linear-reference.png) |
| `build` with in-place camera rotation | [build-rotation-input.mp4](public-demo/build-rotation-input.mp4) | [build-rotation-preview.png](public-demo/build-rotation-preview.png) | [build-rotation-reference.png](public-demo/build-rotation-reference.png) |
| `unwrap` for orbital capture around one cylinder-like object | [unwrap-cylinder-input.mp4](public-demo/unwrap-cylinder-input.mp4) | [unwrap-cylinder-preview.png](public-demo/unwrap-cylinder-preview.png) | [unwrap-cylinder-reference.png](public-demo/unwrap-cylinder-reference.png) |

Overview image: [overview.png](public-demo/overview.png)

## Why These Assets Exist

They solve three problems for the public repository:

* README can show concrete inputs and expected outputs without using confidential videos.
* Contributors can discuss supported scenarios with the same shareable fixtures.
* Releases and GitHub posts can point to public artifacts instead of abstract CLI examples.

## Regenerating

```bash
python3 scripts/generate_public_demo_assets.py
```

The generator is deterministic enough for documentation updates and intentionally keeps files small.

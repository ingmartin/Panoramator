# Public Roadmap

This roadmap is written for repository visitors. It explains what is already stable enough to try, what is intentionally out of scope, and what would most improve the project's public value.

## Current Positioning

* `build` is for scene panoramas from roughly linear motion or in-place rotation.
* `unwrap` is for orbital capture around one object when the expected result is a surface map, not a scene panorama.
* Orbital capture is intentionally not treated as a `build` mode.

## Next Public Improvements

### 1. Stronger demo story

* add rendered before/after images directly into `README.md`
* add short GIF versions of the public demo pack
* add one comparison section for `build` versus generic stitching tools

### 2. Better reproducible examples

* add `examples/` with ready-to-run config files
* add one documented command per scenario: `linear`, `rotation`, `unwrap-cylinder`, `unwrap-curved`
* add a tiny "expected diagnostics" section for each example

### 3. Better trust signals

* publish a concise benchmark table with successes and known failure modes
* document unsupported capture patterns more explicitly
* keep release notes visual, with one image or video per release

### 4. Better contributor entry points

* mark a few issues as `good first issue`
* document architecture decisions around frame selection, projection, and unwrap
* add a public fixture policy for synthetic versus confidential videos

## Explicit Non-Goals

* treating orbital object capture as another `build` panorama mode
* pretending private acceptance videos can be open-sourced
* hiding known capture limitations behind over-broad marketing language

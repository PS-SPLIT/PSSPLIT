# Third-party notices

PS-SPLIT builds on the code of the following projects. Files adapted from them
keep the copyright and licence of the original project. The MIT licence text is
reproduced below once; the Apache-2.0 text is in `LICENSES/Apache-2.0.txt`.

| Project | Licence | Copyright | Used for (in `submission/`) |
|---|---|---|---|
| [Generalized Category Discovery](https://github.com/sgvaze/generalized-category-discovery) | MIT | Copyright (c) 2022 sgvaze | GCD training (`methods/gcd_training/`, `methods/gcd_hier_training/`); also the original source of much of the SelEx code base below |
| [SimGCD](https://github.com/CVMI-Lab/SimGCD) | MIT | Copyright (c) 2022 Xin Wen | `methods/simgcd_training/`, `methods/simgcd_hier_training/` |
| [UNO](https://github.com/DonkeyShot21/UNO) | MIT | Copyright (c) 2021 Enrico Fini | `methods/uno_training/`, `methods/uno_hier_training/` |
| [SelEx](https://github.com/SarahRastegar/SelEx) | MIT | Copyright (c) 2024 Sarah Rastegar | This repository started as a fork of SelEx: `config.py`, `data/` (dataset loaders, augmentations), `models/`, `project_utils/`, `methods/clustering/`, `methods/estimate_k/estimate_k.py`, `methods/contrastive_training/`, `methods/contrastive_hier_training/` |
| [DINO](https://github.com/facebookresearch/dino) | Apache-2.0 | Copyright (c) Facebook, Inc. and its affiliates | `models/vision_transformer.py` |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Apache-2.0 | Copyright (c) Meta Platforms, Inc. and affiliates | `models/vision_transformer2.py` |

The per-file attribution above is at directory level; individual files may mix
code from several of these projects and new code written for PS-SPLIT.

Pretrained weights (DINOv2 ViT-B/14 with registers, DINO ViT-B/16) are not
distributed here; download them from the upstream projects under their terms.

## MIT License (applies to the MIT-licensed projects listed above, with their copyright lines)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

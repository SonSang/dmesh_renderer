This repository contains rendering code for DMesh. The code is based on the implementation of [3D Gaussian Splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization). There are two different renderers here.

* `tri renderer`: It renders a set of (semi-transparent) triangles efficiently. However, it does not support precise depth testing.
* `tet renderer`: It renders a set of (semi-transparent) triangle faces of tetrahedra, which compactly tessellate the entire domain. It supports precise depth testing, but only produces gradients for face opacities.
#include <torch/extension.h>
#include "render.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // tri renderer
    m.def("render_tris", &RasterizeTrianglesCUDA);
    m.def("render_tris_backward", &RasterizeTrianglesBackwardCUDA);

    // tet renderer
    m.def("render_tets", &RenderFTetsCUDA);
    m.def("render_tets_backward", &RenderFTetsBackwardCUDA);
}
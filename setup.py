from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="dmesh_renderer",
    packages=['dmesh_renderer'],
    ext_modules=[
        CUDAExtension(
            name="dmesh_renderer._C",
            sources=[
            # tri renderer
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",

            # tet renderer
            "cuda_renderer/renderer_impl.cu",
            "cuda_renderer/forward.cu",
            "cuda_renderer/backward.cu",

            "render.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
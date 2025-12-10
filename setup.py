import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def make_cuda_ext(
    name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_90,code=sm_90",  # For NVIDIA GH200 and H100
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="coopdet3d",
        packages=find_packages(),
        include_package_data=True,
        package_data={"coopdet3d.ops": ["*/*.so"]},
        install_requires=[
            "mmengine>=0.10.0",
            "mmcv>=2.0.0",
            "mmdet3d>=1.0.0",
            "mmdet>=3.0.0",
            "torch>=1.9.0",
            "torchvision",
            "numpy",
            "scipy",
            "pillow",
            "tqdm",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        license="Apache License 2.0",
        ext_modules=[
            # Only keep unique coopdet3d CUDA ops
            # bev_pool - might be unique for BEVFusion
            make_cuda_ext(
                name="bev_pool_ext",
                module="coopdet3d.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            # paconv - unique PAConv operation
            make_cuda_ext(
                name="assign_score_withk_ext",
                module="coopdet3d.ops.paconv",
                sources=["src/assign_score_withk.cpp"],
                sources_cuda=["src/assign_score_withk_cuda.cu"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )

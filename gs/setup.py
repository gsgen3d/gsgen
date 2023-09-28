import os
from setuptools import setup
import warnings
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    "-O3",
    "-std=c++14",
    # "-G",
    # "-g",
    # "-lineinfo",
    # "-U__CUDA_NO_HALF_OPERATORS__",
    # "-U__CUDA_NO_HALF_CONVERSIONS__",
    # "-U__CUDA_NO_HALF2_OPERATORS__",
    # "-arch",
    # "compute_86",
]

cub_home = os.environ.get("CUB_HOME", None)
if cub_home is None:
    prefix = os.environ.get("CONDA_PREFIX", None)
    if prefix is not None and os.path.isdir(prefix + "/include/cub"):
        cub_home = prefix + "/include"

include_dirs = torch.utils.cpp_extension.include_paths()
print(f"cub_home: {cub_home}")

if cub_home is None:
    warnings.warn(
        "The environment variable `CUB_HOME` was not found."
        "Installation will fail if your system CUDA toolkit version is less than 11."
        "NVIDIA CUB can be downloaded "
        "from `https://github.com/NVIDIA/cub/releases`. You can unpack "
        "it to a location of your choice and set the environment variable "
        "`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
    )
else:
    # include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))
    pass

if os.name == "posix":
    c_flags = ["-O3", "-std=c++14", "-g"]
elif os.name == "nt":
    c_flags = ["/O2", "/std:c++17"]

    # find cl.exe
    def find_cl_path():
        import glob

        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(
                glob.glob(
                    r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"
                    % edition
                ),
                reverse=True,
            )
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError(
                "Could not locate a supported Microsoft Visual C++ installation"
            )
        os.environ["PATH"] += ";" + cl_path

include_dirs.append(os.path.join(_src_path, "src", "include"))

setup(
    name="gs",  # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name="_gs",  # extension name, import this to use CUDA API
            sources=[
                os.path.join(_src_path, "src", f)
                for f in [
                    "render.cu",
                    "bindings.cpp",
                ]
            ],
            extra_compile_args={
                "cxx": c_flags,
                "nvcc": nvcc_flags,
            },
            include_dirs=include_dirs,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)

import os
import shutil
from setuptools import setup, find_packages
import torch
from torch.utils import cpp_extension
from git import Repo
from loguru import logger


class PreparationContext:
    def __init__(self, project_name="mmoe", version="0.1", features=[]):
        self.project_path = os.path.dirname(__file__)
        self.cur_repo = Repo(os.path.dirname(__file__))
        self.project_name = project_name
        self.version = version + "+" + ".".join(
            [
                version,
                self.cur_repo.head.commit.hexsha[:7],
                self.get_device_tag(),
                str(self.get_device_arch()),
            ]
        )
        self.cutlass_path = os.environ.get(
            "CUTLASS_PATH", os.path.dirname(__file__) + "/cutlass"
        )
        if len(features) == 0:
            self.features = ["build_cuda_sm80_grouped_moe_gemv_Wf8_Af16_Of16_Accf32"]
        self.ext_modules = []

        logger.info("Preparing the build environment.")
        self.check_ninja()
        if self.get_device_tag() == "cuda":
            self.prepare_cutlass(self.cutlass_path)

        logger.info(f"Project name: {self.project_name}")
        logger.info(f"Project version: {self.version}")
        logger.info(f"Project path: {self.project_path}")
        logger.info(f"Cutlass path: {self.cutlass_path}")

        logger.info(f"Adding features to the build. Total {len(self.features)} features.")
        for featname in self.features:
            logger.info(f"Adding feature {featname}.")
            if hasattr(self, featname):
                module = getattr(self, featname)()
                if module is not None:
                    self.ext_modules.append(module)
            else:
                logger.warning(f"Feature {featname} is not supported.")

    def get_device_arch(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
        else:
            return 0

    def get_device_tag(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            if "ROCm" in torch.version.cuda:
                return "rocm"
            else:
                return "cuda"
        else:
            return "cpu"

    def check_ninja(self):
        if shutil.which("ninja") is None:
            raise RuntimeError("The ninja is not found. ")

    def prepare_cutlass(self, cutlass_path):
        if not os.path.exists(cutlass_path):
            logger.info("Cloning cutlass repository.")
            try:
                Repo.clone_from("https://github.com/NVIDIA/cutlass.git", cutlass_path)
                logger.info(f"Repository cloned successfully to {cutlass_path}")
            except Exception as e:
                logger.info(f"Failed to clone repository: {e}")
        else:
            logger.info(f"Cutlass repository already exists at {cutlass_path}")

    def build_cuda_sm80_grouped_moe_gemv_Wf8_Af16_Of16_Accf32(self):
        if "cuda-80" not in self.version:
            logger.warning("The current device is not compatible with the feature.")
            return None
        return cpp_extension.CUDAExtension(
                name=f"{self.project_name}.moe.cuda.sm80.grouped_moe_gemv_Wf8_Af16_Of16_Accf32",
                sources=[
                    f"{self.project_name}/moe/cuda_sm80/grouped_moe_gemv_Wf8_Af16_Of16_Accf32.cu",
                ],
                include_dirs=[
                    f"{self.project_name}/moe/cuda/",
                    os.path.join(self.cutlass_path, "tools/util/include"),
                    os.path.join(self.cutlass_path, "include"),
                ],
                extra_link_args=[
                    "-lcuda",
                    "-lculibos",
                    "-lcudart",
                    "-lcudart_static",
                    "-lrt",
                    "-lpthread",
                    "-ldl",
                    "-L/usr/lib/x86_64-linux-gnu/",
                ],
                extra_compile_args={
                    "cxx": ["-std=c++17", "-O3"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-DCUDA_ARCH=80",
                        "-gencode=arch=compute_80,code=compute_80",
                    ],
                },
            )

ctx = PreparationContext()

setup(
    name=f"{ctx.project_name}",
    version=f"{ctx.version}",
    author="Wenxiang@Microsoft Research",
    ext_modules=ctx.ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
    packages=find_packages(exclude=["notebook", "scripts", "test"]),
)

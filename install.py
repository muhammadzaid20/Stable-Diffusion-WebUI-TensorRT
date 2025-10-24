import os
import sys

import launch

python = sys.executable


def _install_optional(package: str, pip_args: str, description: str):
    if not launch.is_installed(package):
        print(f"{description} is not installed! Installing...")
        launch.run_pip(pip_args, package, live=True)


def install():
    # Ensure metadata utilities are available across Python versions.
    if not launch.is_installed("importlib_metadata"):
        launch.run_pip("install importlib_metadata", "importlib_metadata", live=True)

    # packaging is part of the default Automatic1111 environment, but ensure it exists
    # to keep the installer resilient in custom deployments.
    if not launch.is_installed("packaging"):
        launch.run_pip("install packaging", "packaging", live=True)

    from importlib_metadata import PackageNotFoundError, version
    from packaging.version import InvalidVersion, Version

    min_trt_version = Version(os.environ.get("TENSORRT_MIN_VERSION", "9.0.0"))
    preferred_trt_version = os.environ.get("TENSORRT_PIP_VERSION", "10.13.3.9")
    trt_package = os.environ.get("TENSORRT_PIP_PACKAGE", "tensorrt")
    trt_extra_index = os.environ.get(
        "TENSORRT_PIP_EXTRA_INDEX_URL", "https://pypi.nvidia.com"
    )

    cudnn_package = os.environ.get("TENSORRT_CUDNN_PACKAGE", "nvidia-cudnn-cu12")
    cudnn_version = os.environ.get("TENSORRT_CUDNN_VERSION", "9.14.0.64")

    def get_version(pkg_name: str):
        try:
            return Version(version(pkg_name))
        except (PackageNotFoundError, InvalidVersion):
            return None

    installed_trt_version = get_version(trt_package)
    if installed_trt_version and installed_trt_version < min_trt_version:
        launch.run(
            [python, "-m", "pip", "uninstall", "-y", trt_package],
            "removing outdated TensorRT",
        )
        installed_trt_version = None

    if not installed_trt_version:
        print("TensorRT is not installed or out of date. Installing...")
        bootstrap_cudnn = bool(cudnn_package)

        if bootstrap_cudnn:
            cudnn_spec = cudnn_package
            if cudnn_version:
                cudnn_spec += f"=={cudnn_version}"
            launch.run_pip(
                f"install {cudnn_spec} --no-cache-dir",
                cudnn_package,
            )

        trt_spec = trt_package
        if preferred_trt_version:
            trt_spec += f"=={preferred_trt_version}"

        trt_install_cmd = [
            "install",
            trt_spec,
            "--extra-index-url",
            trt_extra_index,
            "--no-cache-dir",
        ]
        launch.run_pip(" ".join(trt_install_cmd), trt_package, live=True)

        if bootstrap_cudnn:
            launch.run(
                [python, "-m", "pip", "uninstall", "-y", cudnn_package],
                f"removing temporary {cudnn_package}",
            )

    installed_cudnn_version = get_version(cudnn_package)
    if installed_cudnn_version and cudnn_version and str(installed_cudnn_version) == cudnn_version:
        launch.run(
            [python, "-m", "pip", "uninstall", "-y", cudnn_package],
            f"removing bootstrap {cudnn_package}",
        )

    _install_optional(
        "polygraphy",
        "install polygraphy --extra-index-url https://pypi.ngc.nvidia.com",
        "Polygraphy",
    )

    if not launch.is_installed("onnx_graphsurgeon"):
        print("GS is not installed! Installing...")
        launch.run_pip("install protobuf==3.20.2", "protobuf", live=True)
        launch.run_pip(
            "install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com",
            "onnx-graphsurgeon",
            live=True,
        )

    _install_optional("optimum", "install optimum", "Optimum")


install()

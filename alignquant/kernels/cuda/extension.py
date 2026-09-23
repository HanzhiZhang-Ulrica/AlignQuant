"""Build and call the native W4/W8A8 CUDA extension."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import torch


_EXTENSION_NAME = "alignquant_w4w8_a8"
_SOURCE_DIR = Path(__file__).resolve().parent
_DEFAULT_EXTENSION_ROOT = Path.home() / ".cache" / "alignquant" / "torch_extensions"
_SOURCES = (
    _SOURCE_DIR / "bindings.cu",
    _SOURCE_DIR / "linear.cu",
)


def _extension_dir() -> Path:
    configured = os.environ.get("TORCH_EXTENSIONS_DIR")
    if configured:
        path = Path(configured).expanduser()
    else:
        path = _DEFAULT_EXTENSION_ROOT
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(path)
    return path


def _ninja() -> Path:
    environment_bins: list[Path] = []
    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        environment_bins.append(Path(conda_prefix) / "bin")
    environment_bins.append(Path(sys.prefix) / "bin")
    environment_bins.append(Path(sys.executable).parent)

    executable: Path | None = None
    for environment_bin in environment_bins:
        environment_ninja = environment_bin / "ninja"
        if environment_ninja.is_file() and os.access(environment_ninja, os.X_OK):
            executable = environment_ninja
            break
    if executable is None:
        discovered = shutil.which("ninja")
        executable = Path(discovered) if discovered is not None else None
    if executable is None:
        raise RuntimeError("The W4/W8A8 CUDA extension requires ninja.")

    # torch.utils.cpp_extension invokes ``ninja`` by name after its own
    # availability check.  Put the verified binary first so PALS cannot pick
    # a user-level Python wrapper whose interpreter lacks the ninja package.
    ninja_bin = str(executable.parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if not path_entries or path_entries[0] != ninja_bin:
        os.environ["PATH"] = ninja_bin + os.pathsep + os.environ.get("PATH", "")
    subprocess.check_output(
        [str(executable), "--version"], text=True, stderr=subprocess.STDOUT
    )
    return executable


def _cutlass_candidates() -> Iterable[Path]:
    for variable in ("ALIGNQUANT_CUTLASS_PATH", "CUTLASS_PATH"):
        if value := os.environ.get(variable):
            yield Path(value)


def _cutlass_include() -> Path:
    tried: list[str] = []
    for root in _cutlass_candidates():
        root = root.expanduser()
        tried.append(str(root))
        for include in (root / "include", root):
            if (include / "cutlass" / "cutlass.h").is_file():
                return include
    raise RuntimeError(
        "CUTLASS headers are required. Set ALIGNQUANT_CUTLASS_PATH to a "
        f"CUTLASS checkout. Tried: {', '.join(tried)}"
    )


def _compiler_version(path: Path) -> tuple[int, ...] | None:
    try:
        identity = subprocess.check_output(
            [str(path), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).lower()
        if "nvc++" in identity or ("g++" not in identity and "gcc" not in identity):
            return None
        raw = subprocess.check_output(
            [str(path), "-dumpfullversion", "-dumpversion"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    values: list[int] = []
    for part in raw.split("."):
        if not part.isdigit():
            break
        values.append(int(part))
    return tuple(values) or None


def _compiler_candidates() -> Iterable[Path]:
    for variable in ("ALIGNQUANT_CXX", "CXX"):
        if value := os.environ.get(variable):
            yield Path(value)
    environment_bin = Path(sys.executable).resolve().parent
    yield environment_bin / "x86_64-conda-linux-gnu-g++"
    yield environment_bin / "g++"
    for name in ("g++-13", "g++-12", "g++-11", "g++-10", "g++-9", "g++"):
        if value := shutil.which(name):
            yield Path(value)


def _compiler() -> Path:
    seen: set[str] = set()
    for candidate in _compiler_candidates():
        candidate = candidate.expanduser()
        if str(candidate) in seen or not candidate.is_file():
            continue
        seen.add(str(candidate))
        version = _compiler_version(candidate)
        if version is not None and version >= (9,):
            return candidate
    raise RuntimeError(
        "The W4/W8A8 CUDA extension requires GCC 9 or newer. Set ALIGNQUANT_CXX."
    )


def _matching_c_compiler(cxx: Path) -> Path | None:
    if "g++" not in cxx.name:
        return None
    candidate = cxx.with_name(cxx.name.replace("g++", "gcc", 1))
    return candidate if candidate.is_file() else None


def _cuda_library_dir(root: Path) -> Path | None:
    for relative in ("lib64", "lib", "targets/x86_64-linux/lib"):
        candidate = root / relative
        if (candidate / "libcudart.so").is_file():
            return candidate
    return None


def _cuda_candidates() -> Iterable[Path]:
    if value := os.environ.get("ALIGNQUANT_CUDA_HOME"):
        yield Path(value)
    version = str(torch.version.cuda or "").strip()
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if value := os.environ.get(variable):
            yield Path(value)
    if nvcc := shutil.which("nvcc"):
        yield Path(nvcc).resolve().parents[1]


def _cuda_home() -> tuple[Path, Path]:
    seen: set[str] = set()
    for root in _cuda_candidates():
        root = root.expanduser()
        if str(root) in seen or not root.is_dir():
            continue
        seen.add(str(root))
        library = _cuda_library_dir(root)
        if (root / "bin" / "nvcc").is_file() and library is not None:
            return root, library
    raise RuntimeError(
        "A complete CUDA toolkit is required. Set ALIGNQUANT_CUDA_HOME to a "
        "directory containing bin/nvcc and libcudart.so."
    )


def _cuda_math_include(cuda_home: Path) -> Path:
    candidates: list[Path] = []
    for variable in ("ALIGNQUANT_CUDA_MATH_INCLUDE", "CUDA_MATH_INCLUDE"):
        if value := os.environ.get(variable):
            candidates.append(Path(value))
    version = (
        cuda_home.name
        if cuda_home.name[:1].isdigit()
        else str(torch.version.cuda or "")
    )
    candidates.extend(
        (cuda_home / "include", cuda_home / "targets/x86_64-linux/include")
    )
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "cusparse.h").is_file():
            return candidate
    raise RuntimeError(
        "CUDA math headers are required. Set ALIGNQUANT_CUDA_MATH_INCLUDE to "
        "the directory containing cusparse.h."
    )


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("The W4/W8A8 extension requires a visible CUDA device.")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 0):
        raise RuntimeError(f"The W4/W8A8 extension requires SM80, got {capability}.")
    missing = [str(source) for source in _SOURCES if not source.is_file()]
    if missing:
        raise RuntimeError(f"Missing CUDA sources: {missing}")

    cutlass = _cutlass_include()
    cxx = _compiler()
    cuda_home, cuda_library = _cuda_home()
    cuda_math = _cuda_math_include(cuda_home)
    os.environ.update(
        CUDA_HOME=str(cuda_home),
        CUDA_PATH=str(cuda_home),
        CXX=str(cxx),
        CUDAHOSTCXX=str(cxx),
    )
    if cc := _matching_c_compiler(cxx):
        os.environ["CC"] = str(cc)
    extension_dir = _extension_dir()
    ninja = _ninja()
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = str(cuda_home)
    return cpp_extension.load(
        name=_EXTENSION_NAME,
        sources=[str(source) for source in _SOURCES],
        extra_include_paths=[str(cutlass), str(cuda_math)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-allow-unsupported-compiler",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_ldflags=[
            f"-L{cuda_library}",
            f"-Wl,-rpath,{cuda_library}",
        ],
        with_cuda=True,
        verbose=False,
    )


def load_extension() -> None:
    """Compile or load the native extension before a timed region."""

    _load_extension()


def alignquant_activation_quantize_cuda(
    activation: torch.Tensor, v_metadata: torch.Tensor, *, tile_major: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply R5 A8 quantization; tile-major output is [M/16,K/64,16,64]."""

    return tuple(
        _load_extension().alignquant_activation_quantize_cuda(
            activation.contiguous(), v_metadata.contiguous(), tile_major
        )
    )


def alignquant_linear_cuda(
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    state_bits: torch.Tensor,
    w4_payload: torch.Tensor,
    w4_scales: torch.Tensor,
    w8_payload: torch.Tensor,
    w8_scales: torch.Tensor,
    w4_row_offsets: torch.Tensor,
    w8_row_offsets: torch.Tensor,
    u_metadata: torch.Tensor,
    *,
    schedule: str = "auto",
    decode_splits: int = 0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run one packed projection; BF16 output is only available on M64 prefill."""

    schedule_codes = {
        "auto": 0,
        "forced_decode": 1,
        "forced_prefill": 2,
        "forced_prefill_m64": 3,
    }
    try:
        schedule_code = schedule_codes[str(schedule)]
    except KeyError as error:
        raise ValueError(
            f"Unknown W4/W8A8 schedule {schedule!r}; expected {tuple(schedule_codes)}."
        ) from error
    if output_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Projection output_dtype must be torch.float32 or torch.bfloat16.")

    return _load_extension().alignquant_linear_cuda(
        activation.contiguous(),
        activation_scales.contiguous(),
        state_bits.contiguous(),
        w4_payload.contiguous(),
        w4_scales.contiguous(),
        w8_payload.contiguous(),
        w8_scales.contiguous(),
        w4_row_offsets.contiguous(),
        w8_row_offsets.contiguous(),
        u_metadata.contiguous(),
        schedule_code,
        decode_splits,
        output_dtype == torch.bfloat16,
    )


__all__ = [
    "load_extension",
    "alignquant_activation_quantize_cuda",
    "alignquant_linear_cuda",
]

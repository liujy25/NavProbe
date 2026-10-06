"""Process-stable source identity for batch result reuse."""
from functools import lru_cache
import hashlib
from importlib.util import find_spec
from importlib import metadata
from pathlib import Path
import platform


@lru_cache(maxsize=1)
def source_digest() -> str:
    digest = hashlib.sha256()
    # Resolve a concrete module: editable installs can expose the package root
    # as a namespace with no __file__. Discovery does not load the geometry.
    geometry_root = Path(find_spec("frontier_exploration.frontier_detection").origin).resolve().parent
    roots = (
        ("navprobe", Path(__file__).resolve().parents[1]),
        ("frontier_exploration", geometry_root),
    )
    for package_name, root in roots:
        for path in sorted(root.rglob("*.py")):
            digest.update(f"{package_name}/{path.relative_to(root).as_posix()}\0".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=2)
def runtime_dependency_identity(detector: str) -> dict:
    """Record behavior-relevant installed versions without importing native engines."""
    names = (
        "habitat-lab", "habitat-sim", "numpy", "scipy", "scikit-image",
        "opencv-python", "opencv-python-headless", "numba", "Pillow",
        "fastdtw", "omegaconf", "openai", "torch", "torchvision",
        "ultralytics" if detector == "yolo_world" else "groundingdino",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    # Editable detector/Habitat-Lab code can change without a version bump.
    # Hash their Python implementation, excluding datasets, models and binaries.
    sources = {}
    for name in ("habitat", "ultralytics" if detector == "yolo_world" else "groundingdino"):
        spec = find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            sources[name] = None
            continue
        digest = hashlib.sha256()
        for root_name in sorted(spec.submodule_search_locations):
            root = Path(root_name)
            for path in sorted(root.rglob("*.py")):
                digest.update(f"{path.relative_to(root).as_posix()}\0".encode())
                digest.update(path.read_bytes())
        sources[name] = digest.hexdigest()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": versions,
        "package_sources": sources,
    }

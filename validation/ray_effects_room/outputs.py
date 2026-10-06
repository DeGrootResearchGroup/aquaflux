"""What an ``aquaflux run`` of a room case wrote, read back as numpy arrays.

A run directory holds ``fields.vtu`` (cell fields, in the mesh's cell order), ``patches.vtm`` with
``patches/<patch>.vtp`` (face fields, in the patch's own face order -- OpenFOAM's), and ``run.yaml``
(the record, with the lamp power and where it went under ``results``). This reads the VTK XML files
directly -- the array tags and the raw appended block -- so the comparison needs nothing beyond numpy
and the YAML parser, and does not depend on the writer it is checking.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RUNS = HERE / "work" / "cases"

_TYPES = {
    "Float64": np.float64,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt8": np.uint8,
    "UInt64": np.uint64,
}


def _arrays(path: Path) -> dict[str, np.ndarray]:
    """Every named data array of a VTK XML file, raw-appended or ASCII."""
    document = path.read_bytes()
    marker = b'<AppendedData encoding="raw">\n_'
    split = document.find(marker)
    head = document if split < 0 else document[:split] + b"</VTKFile>"
    root = ET.fromstring(head.decode())
    payload = b"" if split < 0 else document[split + len(marker) :]
    header = np.dtype(_TYPES[root.attrib.get("header_type", "UInt64")])
    out = {}
    for array in root.iter("DataArray"):
        dtype = np.dtype(_TYPES[array.attrib["type"]])
        if array.attrib["format"] == "ascii":
            flat = np.array(array.text.split(), dtype=dtype)
        else:
            at = int(array.attrib["offset"])
            size = int(np.frombuffer(payload, header, count=1, offset=at)[0])
            flat = np.frombuffer(
                payload, dtype, count=size // dtype.itemsize, offset=at + header.itemsize
            )
        components = int(array.attrib.get("NumberOfComponents", 1))
        out[array.attrib["Name"]] = flat.reshape(-1, components) if components > 1 else flat
    return out


def run_directory(case: str) -> Path:
    """Where the named case file's run wrote."""
    return RUNS / case


def exists(case: str) -> bool:
    return (run_directory(case) / "run.yaml").is_file()


def record(case: str) -> dict:
    """The run's ``run.yaml``."""
    return yaml.safe_load((run_directory(case) / "run.yaml").read_text())


def cell_field(case: str, name: str) -> np.ndarray:
    """A cell field from ``fields.vtu``, one value per cell in the mesh's order."""
    return np.array(_arrays(run_directory(case) / "fields.vtu")[name])


def patch_field(case: str, patch: str, name: str) -> np.ndarray:
    """A face field of one patch, one value per face in the patch's own order."""
    return np.array(_arrays(run_directory(case) / "patches" / f"{patch}.vtp")[name])

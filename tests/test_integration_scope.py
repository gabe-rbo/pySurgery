import numpy as np
import pytest
import builtins

from pysurgery.integrations import trimesh_bridge, pytorch_geometric_bridge
from pysurgery.core.complexes import SimplicialComplex


class _FakeTensor:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakePyGData:
    def __init__(self, edge_index, num_nodes, face=None):
        self.edge_index = _FakeTensor(edge_index)
        self.num_nodes = num_nodes
        self.face = face


def test_trimesh_scope_import_guard():
    if not trimesh_bridge.HAS_TRIMESH:
        with pytest.raises(ImportError):
            trimesh_bridge.trimesh_to_cw_complex(object())


def test_pyg_scope_rejects_face_data(monkeypatch):
    monkeypatch.setattr(pytorch_geometric_bridge, "HAS_TORCH", True)
    # Complete undirected triangle (both directions for each edge)
    edge_index = np.array(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
        dtype=np.int64,
    )
    face = np.array([[0], [1], [2]], dtype=np.int64)
    data = _FakePyGData(edge_index=edge_index, num_nodes=3, face=face)
    cw = pytorch_geometric_bridge.pyg_to_cw_complex(data)
    assert cw.cells[2] == 1


def test_pyg_scope_edge_index_shape_check(monkeypatch):
    monkeypatch.setattr(pytorch_geometric_bridge, "HAS_TORCH", True)
    edge_index = np.array([0, 1, 1, 0], dtype=np.int64)
    data = _FakePyGData(edge_index=edge_index, num_nodes=2)
    with pytest.raises(ValueError):
        pytorch_geometric_bridge.pyg_to_cw_complex(data)


def test_simplicial_to_gudhi_import_guard(monkeypatch):
    sc = SimplicialComplex.from_maximal_simplices([(0, 1, 2)])
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "gudhi":
            raise ImportError("missing gudhi")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError):
        sc.to_gudhi_simplex_tree()


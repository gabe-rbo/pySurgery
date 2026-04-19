import re
with open("pysurgery/bridge/julia_bridge.py", "r") as f:
    text = f.read()

text = re.sub(r'    def compute_circumradius_sq_3d.*', '', text, flags=re.DOTALL)

with open("pysurgery/bridge/julia_bridge.py", "w") as f:
    f.write(text)
    f.write('''    def compute_circumradius_sq_3d(self, points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError("Julia backend unavailable.")
        try:
            res = self.backend.compute_circumradius_sq_3d(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64) + 1 # 1-based for Julia
            )
            return np.array(res, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"compute_circumradius_sq_3d failed: {e!r}")

    def compute_circumradius_sq_2d(self, points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError("Julia backend unavailable.")
        try:
            res = self.backend.compute_circumradius_sq_2d(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64) + 1 # 1-based for Julia
            )
            return np.array(res, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"compute_circumradius_sq_2d failed: {e!r}")

    def quick_mapper_jl(self, G_raw: dict, max_loops: int = 1, min_modularity_gain: float = 1e-6) -> tuple[dict, dict]:
        """
        Executes the high-performance QuickMapper algorithm in Julia.
        G_raw must be a dict with keys "V" (list of ints) and "E" (list of tuples of ints).
        Returns a simplified graph dict and a mapping dictionary L.
        """
        self.require_julia()
        try:
            G_simple, L_jl = self.backend.quick_mapper_jl(
                G_raw,
                int(max_loops),
                float(min_modularity_gain)
            )
            from juliacall import convert
            L_py = convert(dict, L_jl)
            G_simple_py = convert(dict, G_simple)
            return G_simple_py, L_py
        except Exception as e:
            raise RuntimeError(f"quick_mapper_jl failed: {e!r}")

# Singleton instance
julia_engine = JuliaBridge()
''')

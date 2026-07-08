import os
import threading
import importlib.util
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp

# CRITICAL FIX for Segmentation Faults:
# Set this environment variable before any attempt to load or locate `juliacall`.
# This ensures that when juliacall initializes the C-level libjulia runtime,
# the required signal handlers are correctly bound for multi-threaded execution.
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

HAS_JULIACALL = importlib.util.find_spec("juliacall") is not None


class JuliaBridge:
    """Zero-Copy Bridge to execute high-performance Julia algebraic topology operations.

    Overview:
        The JuliaBridge provides a high-performance, zero-copy interface between Python 
        and Julia for heavy algebraic topology computations. It replaces slower 
        subprocess-based alternatives with native memory sharing via `juliacall`, 
        enabling efficient execution of exact algebra, persistent homology, and 
        manifold certification kernels.

    Key Concepts:
        - **Zero-Copy**: Direct memory sharing between NumPy and Julia arrays to avoid serialization overhead.
        - **Singleton**: Process-wide instance ensures Julia runtime is initialized once.
        - **Lazy Initialization**: Julia is only loaded upon the first compute-heavy call.
        - **Warm-up**: Pre-compiles Julia kernels on startup to minimize first-call latency.

    Common Workflows:
        1. **Initialization** -> Automatic via julia_engine.available check.
        2. **Exact Algebra** -> compute_sparse_snf() or compute_sparse_rank_q().
        3. **Geometric Topology** -> is_homology_manifold_jl() or compute_trimesh_boundary_data().
        4. **Optimization** -> compute_optimal_h1_basis_from_simplices().

    Coefficient Ring:
        Supports computations over Z, Q, and Z/pZ depending on the specific backend kernel invoked.

    Attributes:
        error (str | None): Error message if Julia initialization failed.
        jl: The Julia Main module proxy.
        backend: The SurgeryBackend Julia module proxy.
    """

    # Julia packages that surgery_backend.jl `using`/`import`s and must be present
    # for the module to load. (LinearAlgebra/SparseArrays/Statistics/Random are
    # stdlib but listed so a bare environment resolves them explicitly.)
    _REQUIRED_JULIA_PACKAGES = (
        "PythonCall",
        "Combinatorics",
        "PrecompileTools",
        "AbstractAlgebra",
        "IntegerSmithNormalForm",
        "JSON",
        "Statistics",
        "Random",
        "LinearAlgebra",
        "SparseArrays",
        "KrylovKit",
    )
    # Packages used only by optional geometric kernels (graph paths, Delaunay/
    # alpha/crust). Their absence degrades gracefully rather than failing to load.
    _OPTIONAL_JULIA_PACKAGES = (
        "Graphs",
        "SimpleWeightedGraphs",
        "DelaunayTriangulation",
    )
    # Known-good floors for packages where an older-but-already-present
    # version is broken, so a bare `Pkg.add(name)` (a no-op once a package
    # is present at all, regardless of version) would never self-heal.
    # SimpleWeightedGraphs < 1.5 depends on the deprecated LightGraphs.jl
    # (a different UUID from Graphs.jl) instead of Graphs.jl itself, so
    # `SimpleWeightedGraph <: Graphs.AbstractGraph` is false there and
    # optgen_from_simplices's `dijkstra_shortest_paths` call fails with a
    # MethodError. Both `_ensure_julia_packages` and `install_dependencies`
    # enforce these via `_enforce_minimum_versions`, which forces an
    # explicit-version `Pkg.add` (unlike a bare `Pkg.add(name)`, this does
    # upgrade an already-present package) whenever the resolved version is
    # below floor.
    _MIN_JULIA_VERSIONS = {
        "Graphs": "1.14",
        "SimpleWeightedGraphs": "1.5",
    }

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Return a process-wide singleton bridge instance.

        Returns:
            The JuliaBridge singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(JuliaBridge, cls).__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._available = False
                    cls._instance.error = None
                    cls._instance.jl = None
                    cls._instance.backend = None
                    cls._instance._coo_cache = OrderedDict()
                    cls._instance._warmup_level = 0
                    cls._instance._warmup_report = {}
        return cls._instance

    def _initialize(self):
        """Initialize Julia runtime and load backend module lazily.

        This method:
        1. Sets environment variables (JULIA_NUM_THREADS, signal handling)
        2. Loads the Julia runtime via juliacall
        3. Auto-installs missing Julia packages (in CI or when enabled)
        4. Loads the SurgeryBackend module
        5. Runs warm-up compilation workloads

        Returns:
            None
        """
        if self._initialized:
            return

        self._available = False
        if not HAS_JULIACALL:
            self.error = (
                "juliacall is not installed. Install via `pip install juliacall`."
            )
            self._initialized = True
            return

        try:
            # CRITICAL: Set JULIA_NUM_THREADS before juliacall is imported/initialized
            # to ensure multi-threaded Julia loops actually run in parallel.
            if "JULIA_NUM_THREADS" not in os.environ:
                os.environ["JULIA_NUM_THREADS"] = str(os.cpu_count() or 1)

            # CRITICAL FIX for Segmentation Faults:
            # When JULIA_NUM_THREADS > 1, juliacall requires explicit signal handling 
            # orchestration to prevent hard SIGSEGV crashes on multi-threaded executions.
            if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
                os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

            from juliacall import Main as jl_main

            self.jl = jl_main
            
            # CRITICAL: Install missing packages BEFORE loading backend module.
            # This prevents LoadError when surgery_backend.jl tries to 'using Combinatorics' etc.
            in_ci = os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}
            if in_ci:
                self._ensure_julia_packages(verbose=True)
            else:
                self._ensure_julia_packages(verbose=False)
            
            # Prefer the precompiled SurgeryBackend package. It is dev-developed
            # into the juliacall-managed environment by juliapkg (see
            # juliapkg.json) and precompiled once, in a separate pure-Julia
            # subprocess, where its PrecompileTools.@compile_workload block runs
            # and caches native code (pkgimages) to disk. Loading it here just
            # mmaps that cache, so this GIL-holding process never performs the
            # back-to-back first-time JIT compilations that previously deadlocked
            # warmup() (Julia JIT vs. Python GIL; see the warmup deadlock report).
            #
            # If the package cannot be loaded for any reason (juliapkg not yet
            # resolved, a missing optional geometric dependency blocking
            # precompilation, an uninstalled source checkout, ...), fall back to
            # include()ing the source directly. That path reproduces the
            # pre-package behavior exactly -- including graceful degradation when
            # optional packages are absent -- so this change can only add the
            # fast, deadlock-free path, never remove the old one.
            backend_script = os.path.join(
                os.path.dirname(__file__),
                "SurgeryBackend",
                "src",
                "SurgeryBackend.jl",
            )
            try:
                self.jl.seval("import SurgeryBackend")
            except Exception as pkg_err:
                warnings.warn(
                    "Precompiled SurgeryBackend package unavailable "
                    f"({pkg_err!r}); falling back to include()ing the source. "
                    "Warm-up will JIT-compile kernels in-process.",
                    stacklevel=2,
                )
                self.jl.include(backend_script)
            self.jl.seval("import SparseArrays")
            self.backend = self.jl.SurgeryBackend
            self._available = True
            # Mark initialized so require_julia() checks
            # do not recursively re-enter _initialize().
            self._initialized = True
        except Exception as e:
            self.error = f"Failed to initialize Julia backend: {e!r}"
            self._available = False
            self._initialized = True
        finally:
            self._initialized = True

    def _ensure_julia_packages(self, verbose: bool = False) -> None:
        """Install missing Julia packages in the active juliacall environment.

        Automatically ensures all required packages are installed whenever:
        1. Running in CI (CI=true in environment)
        2. Explicitly enabled with PYSURGERY_JULIA_AUTO_INSTALL=1
        3. Running for the first time (PYSURGERY_JULIA_SKIP_INSTALL != 1)
        
        This prevents LoadError failures when surgery_backend.jl tries to import
        packages that aren't installed in the Julia environment.
        
        Args:
            verbose: If True, print status messages during installation.
        """
        # Respect explicit disable flag (for testing or locked environments)
        if os.getenv("PYSURGERY_JULIA_SKIP_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
            if verbose:
                print("[pySurgery] Julia package auto-install disabled (PYSURGERY_JULIA_SKIP_INSTALL=1)")
            return

        # Automatic install enabled in CI or when explicitly requested
        auto_install_ci = os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}
        auto_install_explicit = os.getenv("PYSURGERY_JULIA_AUTO_INSTALL", "").strip().lower() in {"1", "true", "yes"}
        
        if not (auto_install_ci or auto_install_explicit):
            if verbose:
                print("[pySurgery] Julia package auto-install disabled (set CI=true or PYSURGERY_JULIA_AUTO_INSTALL=1 to enable)")
            return

        if verbose:
            print("[pySurgery] Checking Julia package dependencies...")

        # Core packages required for surgery_backend.jl to load
        required_packages = list(self._REQUIRED_JULIA_PACKAGES)
        # Optional packages for geometric kernels (don't fail if missing)
        optional_packages = list(self._OPTIONAL_JULIA_PACKAGES)

        # Detect missing packages
        missing_required = []
        missing_optional = []
        
        for pkg in required_packages:
            try:
                is_missing = bool(self.jl.eval(f'Base.find_package("{pkg}") === nothing'))
                if is_missing:
                    missing_required.append(pkg)
            except Exception:
                # If detection fails, assume missing and try to install
                missing_required.append(pkg)
        
        for pkg in optional_packages:
            try:
                is_missing = bool(self.jl.eval(f'Base.find_package("{pkg}") === nothing'))
                if is_missing:
                    missing_optional.append(pkg)
            except Exception:
                pass  # Silent fail for optional packages

        # Install required packages
        if missing_required:
            if verbose:
                print(f"[pySurgery] Installing missing required packages: {', '.join(missing_required)}")
            try:
                self.jl.eval("import Pkg")
                pkg_expr = ", ".join(f'\"{pkg}\"' for pkg in missing_required)
                self.jl.eval(f"Pkg.add([{pkg_expr}])")
                if verbose:
                    print("[pySurgery] Successfully installed required packages")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to install required Julia packages {missing_required}: {e!r}. "
                    f"Install manually via: julia -e 'using Pkg; Pkg.add({missing_required})'"
                )

        # Install optional packages (best-effort)
        if missing_optional:
            if verbose:
                print(f"[pySurgery] Attempting to install optional packages: {', '.join(missing_optional)}")
            try:
                self.jl.eval("import Pkg")
                pkg_expr = ", ".join(f'\"{pkg}\"' for pkg in missing_optional)
                self.jl.eval(f"Pkg.add([{pkg_expr}])")
                if verbose:
                    print("[pySurgery] Successfully installed optional packages")
            except Exception as e:
                if verbose:
                    print(f"[pySurgery] Optional packages failed (non-critical): {e!r}")

        # Force-upgrade any already-present package below its known-good
        # floor (see _MIN_JULIA_VERSIONS). Detection above only catches
        # packages that are entirely absent (`find_package === nothing`) --
        # an old-but-present version is invisible to it, so a stale
        # environment resolved before a floor was raised would otherwise
        # never self-heal.
        try:
            self.jl.eval("import Pkg")
        except Exception:
            pass
        version_report = self._enforce_minimum_versions(
            self.jl, required_packages + optional_packages, verbose=verbose
        )

        # Ensure the local SurgeryBackend package is developed in the Julia environment
        developed_backend = False
        backend_dir = os.path.join(
            os.path.dirname(__file__),
            "SurgeryBackend",
        )
        if os.path.isdir(backend_dir):
            try:
                is_backend_missing = bool(self.jl.eval('Base.find_package("SurgeryBackend") === nothing'))
            except Exception:
                is_backend_missing = True
            
            if is_backend_missing:
                if verbose:
                    print(f"[pySurgery] Developing local SurgeryBackend package from {backend_dir}...")
                try:
                    self.jl.eval("import Pkg")
                    escaped_path = backend_dir.replace("\\", "\\\\")
                    self.jl.eval(f'Pkg.develop(path="{escaped_path}")')
                    developed_backend = True
                    if verbose:
                        print("[pySurgery] Successfully developed SurgeryBackend package")
                except Exception as e:
                    if verbose:
                        print(f"[pySurgery] Failed to develop SurgeryBackend: {e!r}")

        # Precompile (best-effort, improves startup time)
        if missing_required or missing_optional or version_report["upgraded"] or developed_backend:
            try:
                if verbose:
                    print("[pySurgery] Precompiling Julia packages (may take a minute)...")
                self.jl.eval("Pkg.precompile()")
                if verbose:
                    print("[pySurgery] Precompilation complete")
            except Exception:
                if verbose:
                    print("[pySurgery] Precompilation skipped or failed (non-critical)")
        elif verbose:
            print("[pySurgery] All Julia packages already installed")

    def _enforce_minimum_versions(
        self, jl, packages: List[str], verbose: bool = False
    ) -> Dict[str, Any]:
        """Force-upgrade packages below their floor in `_MIN_JULIA_VERSIONS`.

        `Pkg.add("Name")` (used elsewhere in this module to install anything
        entirely missing) is a no-op when the package is already present,
        regardless of how old that version is. This checks each candidate's
        actually-resolved version and only calls
        `Pkg.add(PackageSpec(name=..., version=...))` -- which does force
        Pkg to re-resolve and upgrade -- for ones genuinely below floor (or
        whose version can't be determined, e.g. a prior install attempt
        failed).

        Args:
            jl: Julia ``Main`` handle with ``Pkg`` already ``import``ed.
            packages: Candidate package names; only ones with an entry in
                `_MIN_JULIA_VERSIONS` are considered, others are skipped.
            verbose: Print progress messages.

        Returns:
            ``{"upgraded": [pkg, ...], "failed": {pkg: err, ...}}``, where
            ``"upgraded"`` lists only packages that actually needed action.
        """
        result: Dict[str, Any] = {"upgraded": [], "failed": {}}
        candidates = [p for p in packages if p in self._MIN_JULIA_VERSIONS]
        if not candidates:
            return result

        installed_versions: Dict[str, str] = {}
        try:
            raw = str(jl.seval(
                'join(["$(info.name)=$(info.version)" for (_, info) in '
                'Pkg.dependencies() if info.version !== nothing], ",")'
            ))
            for entry in raw.split(","):
                name, sep, ver = entry.partition("=")
                if sep:
                    installed_versions[name] = ver
        except Exception:
            pass  # Fall through and attempt every candidate below.

        for pkg in candidates:
            floor = self._MIN_JULIA_VERSIONS[pkg]
            current = installed_versions.get(pkg)
            if current is not None and self._version_ge(current, floor):
                continue
            try:
                jl.seval(
                    f'Pkg.add(Pkg.PackageSpec(name="{pkg}", version="{floor}"))'
                )
                result["upgraded"].append(pkg)
                if verbose:
                    print(f"[pySurgery] Upgraded {pkg} {current or '(unknown)'} -> >={floor}")
            except Exception as exc:
                result["failed"][pkg] = repr(exc)
                if verbose:
                    print(f"[pySurgery] Failed to enforce {pkg} >= {floor}: {exc!r}")
        return result

    @staticmethod
    def _version_ge(version: str, floor: str) -> bool:
        """Compare two dotted version strings numerically, component-wise.

        Args:
            version: The version to check, e.g. ``"1.14.0"``.
            floor: The minimum required version, e.g. ``"1.5"``.

        Returns:
            True if ``version`` is greater than or equal to ``floor``.
        """

        def _parts(v: str) -> tuple:
            out = []
            for p in v.split("."):
                digits = "".join(ch for ch in p if ch.isdigit())
                out.append(int(digits) if digits else 0)
            return tuple(out)

        return _parts(version) >= _parts(floor)

    def install_dependencies(
        self,
        *,
        optional: bool = True,
        precompile: bool = True,
        reinitialize: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Install every Julia package the SurgeryBackend needs, via ``Pkg``.

        Unlike the automatic, environment-gated ``_ensure_julia_packages``, this
        is an explicit, unconditional install of all required packages (and, by
        default, the optional geometric-kernel packages). Use it to provision a
        fresh Julia environment, or to repair one where ``surgery_backend.jl``
        failed to load because a dependency was missing. Works even when the
        backend is currently unavailable, since it only needs the Julia ``Main``.

        Args:
            optional: Also install the optional geometric-kernel packages
                (``Graphs``, ``SimpleWeightedGraphs``, ``DelaunayTriangulation``).
            precompile: Run ``Pkg.precompile()`` after adding packages.
            reinitialize: Reload the backend afterwards so it becomes usable in
                this process without restarting Python.
            verbose: Print progress messages.

        Returns:
            A report dict: ``{"installed": [...], "upgraded": [...],
            "failed": {pkg: err}, "precompiled": bool, "available": bool}``.
            ``"upgraded"`` lists packages that were already present but
            below their floor in ``_MIN_JULIA_VERSIONS`` and were forced to
            a newer version.
        """
        if not HAS_JULIACALL:
            raise RuntimeError(
                "juliacall is not installed. Install it first via `pip install juliacall`."
            )

        # Obtain a Julia Main handle even if the backend itself failed to load.
        jl = self.jl
        if jl is None:
            if "JULIA_NUM_THREADS" not in os.environ:
                os.environ["JULIA_NUM_THREADS"] = str(os.cpu_count() or 1)
            if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
                os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
            from juliacall import Main as jl_main
            jl = self.jl = jl_main

        required = list(self._REQUIRED_JULIA_PACKAGES)
        packages = required + (list(self._OPTIONAL_JULIA_PACKAGES) if optional else [])
        report: dict = {
            "installed": [],
            "upgraded": [],
            "failed": {},
            "precompiled": False,
            "available": False,
        }

        def _add(pkgs: list[str]) -> None:
            """Batch-add; on failure fall back to per-package to isolate the bad one."""
            if not pkgs:
                return
            try:
                expr = ", ".join(f'"{p}"' for p in pkgs)
                jl.seval(f"Pkg.add([{expr}])")
                report["installed"].extend(pkgs)
            except Exception:
                for p in pkgs:
                    try:
                        jl.seval(f'Pkg.add("{p}")')
                        report["installed"].append(p)
                    except Exception as exc:
                        report["failed"][p] = repr(exc)

        with self._lock:
            jl.seval("import Pkg")
            if verbose:
                print(f"[pySurgery] Installing Julia packages: {', '.join(packages)}")
            _add(packages)

            # `_add` above is `Pkg.add([names...])`, a no-op for packages
            # already present -- even below a floor in `_MIN_JULIA_VERSIONS`
            # added after the environment was last resolved. Force those up
            # explicitly so this actually repairs a stale environment
            # rather than only filling in what's entirely missing.
            version_report = self._enforce_minimum_versions(jl, packages, verbose=verbose)
            report["upgraded"] = version_report["upgraded"]
            if version_report["failed"]:
                report["failed"].update(version_report["failed"])

            # Explicitly develop local SurgeryBackend package if it exists
            backend_dir = os.path.join(
                os.path.dirname(__file__),
                "SurgeryBackend",
            )
            if os.path.isdir(backend_dir):
                if verbose:
                    print(f"[pySurgery] Developing local SurgeryBackend package from {backend_dir} ...")
                try:
                    escaped_path = backend_dir.replace("\\", "\\\\")
                    jl.seval(f'Pkg.develop(path="{escaped_path}")')
                    report["installed"].append("SurgeryBackend")
                except Exception as dev_err:
                    report["failed"]["SurgeryBackend"] = repr(dev_err)

            if precompile and not report["failed"]:
                try:
                    if verbose:
                        print("[pySurgery] Pkg.precompile() (may take a minute) ...")
                    jl.seval("Pkg.precompile()")
                    report["precompiled"] = True
                except Exception as exc:
                    report["failed"]["__precompile__"] = repr(exc)

        if reinitialize:
            # Reload the backend now that dependencies are present.
            self._initialized = False
            self._available = False
            self.error = None
            self.backend = None
            self._initialize()
        report["available"] = bool(self._available)

        if verbose:
            if report["failed"]:
                print(f"[pySurgery] Completed with failures: {sorted(report['failed'])}")
            if report["available"]:
                print("[pySurgery] Julia backend is available.")
            elif reinitialize and self.error:
                print(f"[pySurgery] Backend still unavailable: {self.error}")
        return report

    def _minimal_warmup_workloads(self) -> list[tuple[str, callable]]:
        """Return a small workload set that compiles common topology paths.

        Returns:
            A list of (name, workload_callable) tuples for minimal warm-up.
        """
        square_simplices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        square_pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        tetra_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        tetra_pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.5, 0.3, 0.8]],
            dtype=np.float64,
        )
        payload_simplices = [
            (0,),
            (1,),
            (2,),
            (3,),
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 3),
            (0, 1, 2),
            (0, 2, 3),
        ]

        return [
            (
                "h1_opt_square",
                lambda: self.compute_optimal_h1_basis_from_simplices(
                    square_simplices,
                    4,
                    point_cloud=square_pts,
                    max_cycles=4,
                ),
            ),
            (
                "h1_valid_square",
                lambda: self.compute_homology_basis_from_simplices(
                    square_simplices,
                    4,
                    1,
                    point_cloud=square_pts,
                    mode="valid",
                    max_cycles=4,
                ),
            ),
            (
                "h2_tetra_boundary",
                lambda: self.compute_homology_basis_from_simplices(
                    tetra_faces,
                    4,
                    2,
                    point_cloud=tetra_pts,
                    mode="valid",
                    max_cycles=2,
                ),
            ),
            (
                "boundary_payload",
                lambda: self.compute_boundary_payload_from_simplices(
                    payload_simplices,
                    2,
                    include_metadata=False,
                ),
            ),
            (
                "boundary_mod2",
                lambda: self.compute_boundary_mod2_matrix(
                    [(0, 1, 2), (0, 2, 3)],
                    [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)],
                ),
            ),
            (
                "pi1_raw_traces",
                lambda: self.compute_pi1_trace_candidates(
                    np.array([0, 1], dtype=np.int64),
                    np.array([0, 0], dtype=np.int64),
                    np.array([-1, 1], dtype=np.int64),
                    n_vertices=2,
                    n_edges=1,
                ),
            ),
            (
                "metrics_warmup",
                lambda: (
                    self.orthogonal_procrustes(np.eye(2), np.eye(2)),
                    self.pairwise_distance_matrix(np.array([[0., 0.], [1., 1.]]), "euclidean"),
                    self.frechet_distance(np.array([[0., 0.], [1., 1.]]), np.array([[0., 1.], [1., 0.]])),
                    self.gromov_wasserstein_distance(
                        np.array([[0., 1.], [1., 0.]]), np.array([[0., 1.], [1., 0.]]),
                        np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.01, 2
                    ),
                    self.quick_mapper_jl({"V": [0, 1], "E": [(0, 1)]}, 1, -1.0),
                    self.simplify_jl([(0, 1), (1, 2), (0, 2)]),
                    self.compute_cknn_graph(np.array([[0., 0.], [1., 1.]]), 1, 1.0),
                    self.compute_persistence_barcodes({1: sp.eye(2, format="csc")}, "Z2")
                ),
            ),
        ]

    def _full_warmup_workloads(self) -> list[tuple[str, callable]]:
        """Return the extended warm-up workload set for all heavy kernels.

        Returns:
            A list of (name, workload_callable) tuples for full warm-up.
        """

        def _sparse_rank_q_workload():
            import scipy.sparse as sp

            m = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.int64))
            return self.compute_sparse_rank_q(m)

        def _sparse_rank_mod_p_workload():
            import scipy.sparse as sp

            m = sp.csr_matrix(np.array([[1, 1], [0, 1]], dtype=np.int64))
            return self.compute_sparse_rank_mod_p(m, 2)

        def _sparse_cohomology_workload():
            import scipy.sparse as sp

            d_np1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
            d_n = sp.csr_matrix(np.array([[1], [0]], dtype=np.int64))
            return self.compute_sparse_cohomology_basis(d_np1, d_n, cn_size=1)

        def _normal_surface_residual_workload():
            import scipy.sparse as sp

            m = sp.csr_matrix(np.array([[1, -1], [0, 1]], dtype=np.int64))
            v = np.array([[1, 0], [0, 1]], dtype=np.int64)
            return self.compute_normal_surface_residual_norms(m, v)

        return [
            (
                "sparse_snf",
                lambda: self.compute_sparse_snf(
                    np.array([0, 1], dtype=np.int64),
                    np.array([0, 1], dtype=np.int64),
                    np.array([1, 1], dtype=np.int64),
                    (2, 2),
                ),
            ),
            ("sparse_rank_q", _sparse_rank_q_workload),
            ("sparse_rank_mod_p", _sparse_rank_mod_p_workload),
            ("sparse_cohomology_basis", _sparse_cohomology_workload),
            (
                "sparse_cohomology_mod_p",
                lambda: self.compute_sparse_cohomology_basis_mod_p(
                    sp.csr_matrix(np.zeros((1, 0), dtype=np.int64)),
                    sp.csr_matrix(np.array([[1], [0]], dtype=np.int64)),
                    2,
                    cn_size=1
                ),
            ),
            (
                "alexander_whitney_cup",
                lambda: self.compute_alexander_whitney_cup(
                    np.array([1, 0, 1], dtype=np.int64),
                    np.array([1, 1, 0], dtype=np.int64),
                    1,
                    1,
                    [(0, 1, 2)],
                    {(0, 1): 0, (1, 2): 1, (0, 2): 2},
                    {(0, 1): 0, (1, 2): 1, (0, 2): 2},
                    modulus=2,
                ),
            ),
            (
                "group_ring_multiply",
                lambda: self.group_ring_multiply({(0,): 1}, {(0,): 1}, 2),
            ),
            (
                # The twist/clearing persistence reducer: FiltrationReport's hot
                # path once the complex is built. Warm it on a filled triangle so
                # the first real barcode does not pay its JIT cost.
                "filtration_persistence",
                lambda: self.compute_filtration_persistence(
                    np.array([0, 1, 2, 0, 1, 1, 2, 0, 2, 0, 1, 2], dtype=np.int64),
                    np.array([0, 1, 2, 3, 5, 7, 9, 12], dtype=np.int64),
                    np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.float64),
                ),
            ),
            (
                # The fused VR build+filter+reduce path: FiltrationReport's hot path
                # for large Rips clouds. Warm it on a small square so the first real
                # report does not pay its JIT cost.
                "rips_filtration",
                lambda: self.compute_rips_filtration(
                    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                             dtype=np.float64),
                    2.0,
                    2,
                ),
            ),
            (
                # The fused Alpha complex build + Gabriel test + reduce path. Warm it on
                # a small square with Delaunay top simplices.
                "alpha_filtration",
                lambda: self.compute_alpha_filtration(
                    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                             dtype=np.float64),
                    np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
                    2,
                ),
            ),
            (
                # The fused Delaunay-Rips / Delaunay-Cech build + reduce paths, plus
                # their lighter simplices-only counterparts. Same small square with
                # Delaunay top simplices as alpha_filtration above.
                "delaunay_restricted_filtration",
                lambda: (
                    self.compute_delaunay_rips_filtration(
                        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                                 dtype=np.float64),
                        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
                        2,
                    ),
                    self.compute_delaunay_cech_filtration(
                        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                                 dtype=np.float64),
                        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
                        2,
                    ),
                    self.compute_delaunay_rips_simplices(
                        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                                 dtype=np.float64),
                        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
                        2,
                    ),
                    self.compute_delaunay_cech_simplices(
                        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                                 dtype=np.float64),
                        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
                        2,
                    ),
                ),
            ),
            (
                # The implicit-cohomology Rips engine (Phase B), used for high
                # max_dimension. Warm it on a small tetrahedral cloud at max_dim=3
                # so its first real call does not pay the JIT cost.
                "rips_cohomology",
                lambda: self.compute_rips_cohomology(
                    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64),
                    3.0,
                    3,
                ),
            ),
            (
                "pi1_abelianization",
                lambda: self.abelianize_and_bhs_rank(["a"], [["a", "a"]]),
            ),
            (
                "integral_lattice_isometry",
                lambda: self.integral_lattice_isometry(
                    np.array([[1, 0], [0, 1]], dtype=np.int64),
                    np.array([[1, 0], [0, 1]], dtype=np.int64),
                ),
            ),
            (
                "normal_surface_residual_norms",
                _normal_surface_residual_workload,
            ),
            (
                "manifold_certification",
                lambda: self.is_homology_manifold_jl([(0, 1), (1, 2), (0, 2)], 1),
            ),
            (
                "broad_phase_pairs",
                lambda: self.compute_broad_phase_pairs(
                    np.array([[0., 0.], [1., 1.]]), np.array([0.1, 0.1]), tol=0.1
                ),
            ),
            (
                "hermitian_signature",
                lambda: self.compute_hermitian_signature(np.eye(2)),
            ),
            (
                "multisignature",
                lambda: self.compute_multisignature(np.eye(2), 2),
            ),
            (
                "boundary_data_assembly",
                lambda: (
                    self.compute_boundary_data_from_simplices([(0, 1, 2)], 2),
                    self.compute_boundary_payload_from_simplices([(0, 1, 2)], 2),
                    self.compute_trimesh_boundary_data([(0, 1, 2)], 3)
                ),
            ),
            (
                "pi1_data_assembly",
                lambda: self.compute_pi1_raw_data(
                    np.array([0, 1], dtype=np.int64), np.array([0, 0], dtype=np.int64), np.array([-1, 1], dtype=np.int64),
                    2, 1,
                    np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), np.array([1], dtype=np.int64),
                    1
                ),
            ),
            (
                "geometric_kernels",
                lambda: (
                    self.triangulate_surface_delaunay(np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]], dtype=float)),
                    self.enumerate_cliques_sparse(np.array([1, 2, 3], dtype=np.int64), np.array([1, 0], dtype=np.int64), 2, 2),
                    self.compute_circumradius_sq_2d(np.array([[0,0], [1,0], [0,1]], dtype=float), np.array([[0, 1, 2]], dtype=np.int64)),
                    self.compute_circumradius_sq_3d(np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float), np.array([[0, 1, 2, 3]], dtype=np.int64)),
                    # Includes an exactly-singular matrix (duplicate row) so the Rational{BigInt}
                    # exact-fallback tier is compiled here too, not only the float64 filter.
                    self.exact_signs_of_determinants_batch(
                        np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 2.0], [1.0, 2.0]]], dtype=float)
                    ),
                    self.compute_cknn_graph_accelerated(np.array([[0,0], [1,1]], dtype=float), np.array([1.0, 1.0]), 1.0),
                    # Vietoris-Rips builder: the dominant kernel in FiltrationReport's
                    # hot path; compile it here so the first filtration report does
                    # not pay its JIT cost.
                    self.compute_vietoris_rips(np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float), 1.5, 2),
                    self.compute_vietoris_rips_from_distance_matrix(
                        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=float), 1.5, 2
                    ),
                    self.simplify_jl([(0, 1), (1, 2), (0, 2)])
                ),
            ),
            (
                "linking_intersection",
                lambda: (
                    self.linking_intersection_pairing(np.array([1], dtype=np.int64), np.array([1], dtype=np.int64), [[0, 1]], [[0, 1, 2]], 3),
                    self.linking_intersection_batch([np.array([1], dtype=np.int64)], np.array([1], dtype=np.int64), [[0, 1]], [[0, 1, 2]], 3),
                    self.linking_intersect_2chains(np.array([1], dtype=np.int64), np.array([1], dtype=np.int64), [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]])
                )
            ),
            (
                "linking_gauss_riemann",
                # Compile the embedding-based Gauss linking integral against a
                # small Hopf-link-style workload so test runs hit a hot kernel.
                lambda: self.linking_gauss_riemann(
                    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
                    np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64),
                    np.array([1, 1], dtype=np.int64),
                    np.array([[0.5, 0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float64),
                    np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5]], dtype=np.float64),
                    np.array([1, 1], dtype=np.int64),
                    24,
                ),
            ),
            (
                "knot_invariants",
                # Pre-compile Seifert→Alexander and the signature pathway so
                # the very first call doesn't pay an extra JIT cost.
                lambda: (
                    self.alexander_from_seifert(np.array([[-1, 1], [0, -1]], dtype=np.int64)),
                    self.knot_signature(np.array([[-1, 1], [0, -1]], dtype=np.int64)),
                ),
            ),
            (
                "discrete_hodge",
                lambda: (
                    self.compute_hodge_harmonics(sp.csr_matrix(np.eye(2, dtype=np.float64)), 0),
                    self.compute_hodge_decomposition(sp.csr_matrix(np.zeros((1, 2), dtype=np.float64)), sp.csr_matrix(np.zeros((2, 1), dtype=np.float64)), sp.csr_matrix(np.eye(2, dtype=np.float64)), np.array([1.0, 1.0]), 0)
                )
            ),
        ]

    def _coverage_warmup_workloads(self) -> list[tuple[str, callable]]:
        """Return warm-up workloads for the backend kernels not covered elsewhere.

        Completes coverage of the JuliaBridge surface: a full warmup that also runs
        these compiles every ``self.backend`` kernel. Risky/structured kernels are
        isolated in their own entries so a single failure does not block the others
        (warm-up is fault-tolerant).

        Returns:
            A list of (name, workload_callable) tuples for the remaining kernels.
        """
        tetra = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        tetra_max = np.array([[0, 1, 2, 3]], dtype=np.int64)
        grid = np.array([[float(i), float(j), 0.0] for i in range(3) for j in range(3)],
                        dtype=np.float64)

        def _exact_snf_certs():
            m = sp.csr_matrix(np.array([[2, 1], [0, 3]], dtype=np.int64))
            self.compute_batch_snf([sp.csr_matrix(np.eye(2, dtype=np.int64))])
            self.compute_markowitz_column_order(m)
            self.compute_modular_rank_certificate(m)
            self.compute_padic_snf_diagonal(sp.csr_matrix(np.array([[2, 0], [0, 1]], dtype=np.int64)))

        def _alpha_witness_morse():
            self.compute_alpha_threshold_emst(tetra, tetra_max)
            self.compute_alpha_complex_simplices(tetra, tetra_max, 100.0, 2)
            self.compute_witness_complex_simplices(grid, np.array([0, 4, 8], dtype=np.int64), 2.0, 2)
            self.compute_discrete_morse_gradient_jl([[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]])

        def _cohomology_basis():
            d1 = sp.csr_matrix(np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.int64))
            d2 = sp.csr_matrix(np.array([[1], [-1], [1]], dtype=np.int64))
            d3 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
            self.compute_cohomology_basis_jl(d1, d2, d3)

        def _controlled_cohomology_chain():
            # Z/2 = <a | a^2>: exercises Todd-Coxeter, the Cayley table, group-ring
            # convolution, Fox derivatives (real+complex) and the cover lift.
            ok, order, table = self.todd_coxeter_index(["a"], ["a a"], 12)
            cayley, inverse, id_idx, words = self.cayley_table(table, list("a"))
            self.cayley_convolve(np.array([1, 0], dtype=np.int64),
                                 np.array([0, 1], dtype=np.int64), cayley)
            gen_group_idx = 2 if int(id_idx) == 1 else 1   # 'a' is the non-identity element
            # The Fox kernel indexes rho by GROUP element (1..order), so supply one
            # (degree x degree) matrix per group element (trivial rep here).
            rho = [np.array([[1.0]], dtype=np.float64) for _ in range(int(order))]
            for cdtype in (False, True):
                self.fox_derivative_block(
                    [1, 1], 1, cayley, [gen_group_idx], inverse, rho, 1, complex_dtype=cdtype,
                )
            self.lift_boundary_to_cover(
                np.array([1], dtype=np.int64), np.array([1], dtype=np.int64),
                np.array([1], dtype=np.int64), np.array([1], dtype=np.int64),
                int(order), 1, 1, cayley,
            )

        def _surgery_linking():
            self.sphere_recognition_pl(
                [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], 2)   # boundary of a 3-simplex = S^2
            Cq, Cqp1 = [[0, 1], [1, 2], [0, 2]], [[0, 1, 2]]
            B = self.surgery_relative_boundary_sparse(Cq, Cqp1, [0])
            b = np.zeros(len(Cq), dtype=np.int64); b[0] = 1
            self.linking_seifert_solve_z(B, b)

        def _surgery_handle():
            K = {0: [[0], [1], [2]], 1: [[0, 1], [1, 2], [0, 2]], 2: [[0, 1, 2]]}
            self.surgery_handle_attach(K, [[0], [1]], [[0], [1]], [[0, 1]], 3, 1, 2)

        def _crust():
            pts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.6]],
                           dtype=np.float64)
            combined = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 4]], dtype=np.int64)
            self.compute_crust_simplices(pts, combined, 4)

        def _voronoi_poles():
            pts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
            vor_vertices = np.array([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]], dtype=np.float64)
            cell_vertex_lists = [[0, 1], [0], [0], [0]]
            self.compute_voronoi_poles(pts, vor_vertices, cell_vertex_lists)

        def _cocone_filter():
            pts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, -1.0]],
                           dtype=np.float64)
            tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
            normals = np.array([[0., 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
            pole_radius = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            self.compute_cocone_filter(pts, tets, normals, pole_radius, np.deg2rad(40.0), 0.6)

        def _prune_and_walk():
            pts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
                           dtype=np.float64)
            # Two candidate triangles at vertex 0: a clean fan (fast path) plus a
            # 3rd triangle at vertex 1 to also exercise the slow angular-walk path.
            tris = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3]],
                            dtype=np.int64)
            normals = np.array([[0., 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                               dtype=np.float64)
            self.compute_prune_and_walk(pts, tris, normals)

        def _tangential_local_stars():
            # 2 disjoint 4-point neighborhoods (offsets splits them): each is a
            # small square with its own center point first (local index 0/1).
            flat_global_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
            flat_coords = np.array([
                [0., 0], [1, 0], [0, 1], [1, 1],
                [0., 0], [1, 0], [0, 1], [1, 1],
            ], dtype=np.float64)
            offsets = np.array([0, 4, 8], dtype=np.int64)
            self.compute_tangential_local_stars_2d(flat_global_idx, flat_coords, offsets)

        return [
            ("exact_snf_certificates", _exact_snf_certs),
            ("alpha_witness_morse", _alpha_witness_morse),
            ("cohomology_basis_jl", _cohomology_basis),
            ("controlled_cohomology_chain", _controlled_cohomology_chain),
            ("surgery_linking", _surgery_linking),
            ("surgery_handle_attach", _surgery_handle),
            ("crust_simplices", _crust),
            ("voronoi_poles", _voronoi_poles),
            ("cocone_filter", _cocone_filter),
            ("prune_and_walk", _prune_and_walk),
            ("tangential_local_stars", _tangential_local_stars),
        ]

    def compute_normal_surface_residual_norms(
        self,
        matrix,
        coordinate_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute `||A * x_i||_2` in batch for normal-surface coordinate columns.

        What is Being Computed?:
            Computes the L2 norm of the residual vector A*x for each coordinate vector 
            in the input matrix. This is used to verify if normal surface coordinates 
            satisfy the matching equations A*x = 0.

        Algorithm:
            1. Convert the sparse matrix to cached COO triplets.
            2. Pass triplets and the coordinate matrix to the Julia backend.
            3. Perform batched matrix-vector multiplication and norm calculation in Julia.
            4. Return the resulting norms as a NumPy array.

        Preserved Invariants:
            - The residual norm is 0 if and only if the coordinate vector represents a valid normal surface.
            - Linear scaling: ||A * (c*x)|| = |c| * ||A * x||.

        Args:
            matrix: The sparse matrix A (matching equations).
            coordinate_matrix: A 2D array where each column is a coordinate vector x_i.

        Returns:
            np.ndarray: An array of L2 norms for each column in coordinate_matrix.

        Use When:
            - Validating normal surface candidates at scale.
            - Checking convergence of normal surface optimization.
            - Working with large-scale matching equation systems where Python speed is insufficient.

        Example:
            norms = julia_engine.compute_normal_surface_residual_norms(A, coords)
            valid_mask = norms < 1e-9
        """
        self.require_julia()
        coords = np.asarray(coordinate_matrix, dtype=np.int64)
        if coords.ndim != 2:
            raise ValueError("coordinate_matrix must be a 2D array")
        if matrix is None:
            raise ValueError("matrix is required")
        if int(matrix.shape[1]) != int(coords.shape[0]):
            raise ValueError(
                "coordinate_matrix row count must match matrix column count"
            )
        if coords.shape[1] == 0:
            return np.zeros(0, dtype=np.float64)
        if int(matrix.shape[0]) == 0:
            return np.zeros(coords.shape[1], dtype=np.float64)

        if matrix.nnz == 0:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
            vals = np.array([], dtype=np.int64)
        else:
            rows, cols, vals = self._coo_triplets_cached(matrix)

        norms = self.backend.normal_surface_residual_norms(
            rows,
            cols,
            vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            coords,
        )
        return np.asarray(norms, dtype=np.float64)

    def compute_broad_phase_pairs(
        self,
        centroids: np.ndarray,
        radii: np.ndarray,
        *,
        tol: float,
    ) -> np.ndarray:
        """Compute candidate simplex index pairs `(i, j)` with `i < j` via Julia.

        Args:
            centroids: An (N, D) array of simplex centroids.
            radii: An (N,) array of simplex radii.
            tol: Tolerance for overlap detection.

        Returns:
            An (M, 2) array of candidate pairs.

        Raises:
            ValueError: If inputs are invalid or incompatible.
        """
        self.require_julia()
        ctr = np.asarray(centroids, dtype=np.float64)
        rad = np.asarray(radii, dtype=np.float64).reshape(-1)
        if ctr.ndim != 2:
            raise ValueError("centroids must be a 2D array")
        if rad.ndim != 1:
            raise ValueError("radii must be a 1D array")
        if ctr.shape[0] != rad.shape[0]:
            raise ValueError("centroids and radii must have the same number of rows")
        if ctr.shape[0] <= 1:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = self.backend.embedding_broad_phase_pairs(ctr, rad, float(tol))
        out = np.asarray(pairs, dtype=np.int64)
        if out.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return out.reshape(-1, 2)

    def _run_warmup(self, mode: str) -> dict:
        """Execute warm-up workloads and cache the resulting status report.

        Args:
            mode: Warm-up mode ('minimal' or 'full').

        Returns:
            A dictionary containing the warm-up status report.
        """
        target_level = 2 if mode == "full" else 1
        with self._lock:
            if self._warmup_level >= target_level:
                report = dict(self._warmup_report)
                report["cached"] = True
                return report

            if not self._available or self.jl is None or self.backend is None:
                report = {
                    "mode": mode,
                    "available": False,
                    "completed": [],
                    "failed": {},
                    "cached": False,
                }
                self._warmup_report = report
                return report

            workloads = list(self._minimal_warmup_workloads())
            if mode == "full":
                workloads.extend(self._full_warmup_workloads())
                workloads.extend(self._coverage_warmup_workloads())

            report = {
                "mode": mode,
                "available": True,
                "completed": [],
                "failed": {},
                "cached": False,
            }
            # Defer PythonCall's Python-object finalizations for the duration of
            # the warm-up loop. The precompiled SurgeryBackend package moved all
            # heavy kernel compilation into a GIL-free precompile subprocess, but
            # the thin Python<->Julia boundary specializations still JIT here,
            # in-process, under the GIL: juliacall passes zero-copy PyArray/Py
            # argument types that only exist once Python is running, so they
            # cannot be precompiled ahead of time. That JIT is the last place the
            # historic warm-up deadlock could form -- compilation allocates,
            # which triggers a Julia GC, whose PythonCall finalizer hook tries to
            # DecRef queued Python objects, which needs the very GIL the blocked
            # main thread is holding. Disabling PythonCall's GC queues those
            # frees instead of running them under the GIL, removing the deadlock
            # precondition. This is the safe lever (unlike releasing the GIL,
            # which these pyconvert-calling kernels cannot tolerate) and its cost
            # is bounded -- warm-up inputs are tiny, and enable()+gc() drains the
            # queue immediately after. Best-effort: older PythonCall lacks it.
            _pc_gc_disabled = False
            try:
                self.jl.seval("PythonCall.GC.disable()")
                _pc_gc_disabled = True
            except Exception:
                pass
            try:
                for name, workload in workloads:
                    try:
                        print(f"[julia_engine.warmup() :: WARMING UP] {name}")
                        workload()
                        report["completed"].append(name)
                    except Exception as exc:
                        report["failed"][name] = repr(exc)
            finally:
                if _pc_gc_disabled:
                    try:
                        self.jl.seval("PythonCall.GC.enable(); PythonCall.GC.gc()")
                    except Exception:
                        pass

            # Explicitly delete trash variables generated during warmup to free memory
            workloads.clear()
            del workloads
            import gc
            gc.collect()

            self._warmup_level = max(self._warmup_level, target_level)
            self._warmup_report = report
            if report["failed"]:
                warnings.warn(
                    "Julia warm-up completed with partial failures; regular execution will still proceed. "
                    f"Failed workloads: {sorted(report['failed'].keys())}",
                    stacklevel=2,
                )
            return report

    def warmup(self) -> dict:
        """Fully warm up Julia-backed bridge paths (best-effort, non-fatal).

        Returns:
            A dictionary containing the warm-up status report.
        """
        if not self.available:
            return {
                "mode": "full",
                "available": False,
                "completed": [],
                "failed": {"initialize": self.error or "Julia unavailable"},
                "cached": False,
            }
        return self._run_warmup("full")

    def _ensure_initialized(self):
        """Initialize the bridge on first use.

        Returns:
            None
        """
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._initialize()

    @property
    def available(self) -> bool:
        """Whether Julia backend is initialized and available for compute calls.

        Returns:
            True if available, False otherwise.
        """
        self._ensure_initialized()
        return self._available

    @available.setter
    def available(self, value: bool):
        """Test hook to override availability without reinitializing Julia.

        Args:
            value: The availability status to set.
        """
        # Allow tests to monkeypatch availability while preserving singleton API.
        self._initialized = True
        self._available = bool(value)

    def require_julia(self):
        """Raise a structured error when Julia backend is unavailable.

        Raises:
            SurgeryError: If the Julia backend is unavailable.
        """
        from pysurgery.core.exceptions import SurgeryError

        if not self.available:
            raise SurgeryError(
                f"High-performance exact algebra requires Julia: {self.error}"
            )

    def _coo_cache_key(self, matrix) -> tuple:
        """Build a stable cache key for sparse COO triplet conversion.

        Args:
            matrix: The sparse matrix to build a key for.

        Returns:
            A tuple representing the cache key.
        """
        data_ptr = None
        try:
            data_ptr = int(matrix.data.__array_interface__["data"][0])
        except Exception:
            data_ptr = None
        return (
            id(matrix),
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            int(matrix.nnz),
            data_ptr,
        )

    def _coo_triplets_cached(self, matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached COO triplets `(rows, cols, vals)` for a sparse matrix.

        Args:
            matrix: The sparse matrix to get triplets for.

        Returns:
            A tuple of (rows, cols, vals) arrays.
        """
        key = self._coo_cache_key(matrix)
        cached = self._coo_cache.get(key)
        if cached is not None:
            self._coo_cache.move_to_end(key)
            return cached

        if sp.issparse(matrix) and matrix.format == "coo":
            coo = matrix
        elif sp.issparse(matrix):
            coo = matrix.tocoo()
        else:
            coo = sp.csr_matrix(matrix).tocoo()

        triplets = (
            np.asarray(coo.row, dtype=np.int64),
            np.asarray(coo.col, dtype=np.int64),
            np.asarray(coo.data, dtype=np.int64),
        )
        self._coo_cache[key] = triplets
        if len(self._coo_cache) > 24:
            self._coo_cache.popitem(last=False)
        return triplets

    def _flatten_simplices(self, simplices: list) -> tuple[np.ndarray, np.ndarray]:
        """Flatten ragged simplex lists into `(flat_vertices, offsets)` arrays.

        Args:
            simplices: A list of simplex tuples/lists.

        Returns:
            A tuple of (flat_vertices, offsets) arrays.
        """
        offsets = np.zeros(len(simplices) + 1, dtype=np.int64)
        total = 0
        for i, simplex in enumerate(simplices, start=1):
            total += len(simplex)
            offsets[i] = total
        flat = np.empty(total, dtype=np.int64)
        cursor = 0
        for simplex in simplices:
            slen = len(simplex)
            if slen:
                flat[cursor : cursor + slen] = np.asarray(simplex, dtype=np.int64)
                cursor += slen
        return flat, offsets

    def compute_hermitian_signature(self, matrix_array: np.ndarray) -> int:
        """Compute the signature of a symmetric real matrix via Julia backend.

        Args:
            matrix_array: The symmetric real matrix.

        Returns:
            The topological signature of the matrix.
        """
        self.require_julia()
        # Direct zero-copy passing to Julia via PyArray
        return int(
            self.backend.hermitian_signature(np.asarray(matrix_array, dtype=np.float64))
        )

    def compute_sparse_snf(
        self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple
    ) -> np.ndarray:
        """Executes the highly optimized Julia Sparse SNF backend.

        Args:
            rows: Row indices of the sparse matrix.
            cols: Column indices of the sparse matrix.
            vals: Values of the sparse matrix.
            shape: Shape of the sparse matrix.

        Returns:
            An array containing the diagonal factors of the SNF.
        """
        self.require_julia()
        # Direct zero-copy passing of NumPy arrays
        factors = self.backend.exact_snf_sparse(
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(vals, dtype=np.int64),
            int(shape[0]),
            int(shape[1]),
        )
        return np.array(factors, dtype=np.int64)

    def compute_markowitz_column_order(
        self, matrix
    ) -> np.ndarray:
        """Return a 0-indexed column permutation minimising fill-in (Markowitz criterion).

        What is Being Computed?:
            For each column c the Markowitz score is
            (col_nnz(c)−1) × min_{r:A[r,c]≠0}(row_nnz(r)−1).
            Columns are sorted by score ascending, so unit/singleton columns
            come first and exploit O(V+E) leaf-peeling.

        Algorithm:
            1. Build col_nnz and row_nnz from COO triplets.
            2. For each column find the minimum row_nnz among its non-zeros.
            3. Sort columns by (col_nnz−1) × (min_row_nnz−1).
            4. Return 0-indexed permutation.

        Preserved Invariants:
            Column permutation is unimodular; the SNF diagonal is unchanged.

        Args:
            matrix: Sparse integer matrix (scipy.sparse or similar).

        Returns:
            np.ndarray: 0-indexed column permutation of length n.

        Use When:
            Pre-conditioning a sparse core before calling exact_snf_sparse to
            reduce coefficient growth during elimination.

        Example:
            perm = julia_engine.compute_markowitz_column_order(boundary_matrix)
            A_permuted = boundary_matrix[:, perm]
        """
        self.require_julia()
        if matrix is None or (hasattr(matrix, "nnz") and matrix.nnz == 0):
            n = matrix.shape[1] if matrix is not None else 0
            return np.arange(n, dtype=np.int64)
        rows, cols, vals = self._coo_triplets_cached(matrix)
        perm = self.backend.snf_markowitz_column_order(
            rows, cols, vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        )
        return np.array(perm, dtype=np.int64)

    def compute_modular_rank_certificate(
        self,
        matrix,
        primes: list[int] | None = None,
    ) -> dict:
        """Certify matrix rank via rank(A mod p) for multiple primes.

        What is Being Computed?:
            Computes rank(A mod p) for each prime p.  Because rank(A mod p) =
            #{i : p ∤ d_i} (where d_i are SNF diagonal entries), agreement
            across primes certifies the ℤ-rank with mathematical certainty when
            gcd(product-of-primes, product-of-d_i) = 1 for all i.

        Algorithm:
            1. For each prime p: dense Gaussian elimination over GF(p).
            2. Collect (prime, rank) pairs.
            3. Check agreement; compute lower bound = max of all mod-p ranks.

        Preserved Invariants:
            rank(A mod p) ≤ rank(A over Q) for all p.  Lower bound is tight
            when at least one chosen prime is coprime to all torsion coefficients.

        Args:
            matrix: Sparse integer matrix.
            primes: List of primes to use.  Defaults to {2,3,5,7,11,13}.

        Returns:
            dict with keys:
              primes         – list of primes used
              ranks          – list of mod-p ranks
              all_agree      – bool
              lower_bound    – int (max of ranks)
              certified_rank – int (agreed rank, or -1 if disagreement)
              exact          – bool (True when all_agree)

        Use When:
            - Quickly verifying rank before full SNF computation.
            - Cross-checking an exact SNF result.

        Example:
            cert = julia_engine.compute_modular_rank_certificate(boundary_matrix)
            assert cert["all_agree"], "Rank inconsistency detected"
        """
        self.require_julia()
        if primes is None:
            primes = [2, 3, 5, 7, 11, 13]
        if matrix is None or (hasattr(matrix, "nnz") and matrix.nnz == 0):
            return {
                "primes": primes, "ranks": [0] * len(primes),
                "all_agree": True, "lower_bound": 0,
                "certified_rank": 0, "exact": True,
            }
        rows, cols, vals = self._coo_triplets_cached(matrix)
        primes_arr = np.array(primes, dtype=np.int64)
        res = self.backend.modular_rank_certification_jl(
            rows, cols, vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            primes_arr,
        )
        return {
            "primes":         list(np.array(res.primes, dtype=np.int64)),
            "ranks":          list(np.array(res.ranks,  dtype=np.int64)),
            "all_agree":      bool(res.all_agree),
            "lower_bound":    int(res.lower_bound),
            "certified_rank": int(res.certified_rank),
            "exact":          bool(res.all_agree),
        }

    def compute_padic_snf_diagonal(
        self,
        matrix,
        primes: list[int] | None = None,
        max_e: int = 8,
    ) -> np.ndarray:
        """Reconstruct SNF diagonal via deterministic p-adic CRT lifting.

        What is Being Computed?:
            For each prime p the function builds the rank sequence
            r_e = #{i : v_p(d_i) < e} by Gaussian elimination over ℤ/p^eℤ,
            decodes v_p(d_k), and reconstructs d_k = ∏_p p^{v_p(d_k)}.
            This path does NOT call IntegerSmithNormalForm; it is an independent
            verification route.

        Algorithm:
            1. Compute total rank via floating-point for upper bound.
            2. For each prime p ∈ primes, for e = 1..max_e:
               a. _padic_rank_step(A, p, p^e) → r_e.
               b. Decode v_p(d_k) = (first e with r_e ≥ k) − 1.
            3. d_k = ∏_p p^{v_p(d_k)}; return sorted in non-decreasing order.

        Preserved Invariants:
            Exact when every prime factor of every d_k ≤ max(primes) and
            max_e ≥ max_k v_p(d_k) for all p.

        Args:
            matrix: Sparse integer matrix.
            primes: Primes to use.  Defaults to [2,3,5,7,11,13,17,19,23,29,31].
            max_e:  Maximum p-adic depth per prime (default 8).

        Returns:
            np.ndarray: Non-zero SNF diagonal in non-decreasing order.

        Use When:
            - Cross-validating the exact SNF computed by IntegerSmithNormalForm.
            - Working with matrices whose torsion is known to be small.

        Example:
            diag = julia_engine.compute_padic_snf_diagonal(boundary_2)
            # Cross-check: should match compute_sparse_snf(...)
        """
        self.require_julia()
        if primes is None:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        if matrix is None or (hasattr(matrix, "nnz") and matrix.nnz == 0):
            return np.array([], dtype=np.int64)
        rows, cols, vals = self._coo_triplets_cached(matrix)
        primes_arr = np.array(primes, dtype=np.int64)
        result = self.backend.padic_snf_diagonal_jl(
            rows, cols, vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            primes_arr,
            int(max_e),
        )
        return np.array(result, dtype=np.int64)

    def compute_batch_snf(self, matrices: list) -> list[np.ndarray]:
        """Compute exact SNF for a batch of sparse matrices in parallel (Julia threads).

        What is Being Computed?:
            Calls exact_snf_sparse for each matrix independently, dispatching
            work across Julia thread-pool workers via Threads.@threads.

        Algorithm:
            1. Convert each matrix to COO triplets.
            2. Pass batch arrays to Julia; each worker runs leaf-peel + Markowitz +
               IntegerSmithNormalForm/AbstractAlgebra independently.
            3. Collect and return results in input order.

        Preserved Invariants:
            Each sub-computation is independent; thread safety is guaranteed by
            Julia's task-local state.

        Args:
            matrices: List of sparse integer matrices (scipy.sparse or similar).

        Returns:
            list[np.ndarray]: SNF diagonal for each input matrix, in order.
              Matrices that fail return an empty array.

        Use When:
            - Computing homology for many boundary dimensions simultaneously.
            - Parallelising independent SNF computations on a multi-core machine.

        Example:
            diagonals = julia_engine.compute_batch_snf([d1, d2, d3])
        """
        self.require_julia()
        if not matrices:
            return []
        batch_rows, batch_cols, batch_vals = [], [], []
        batch_m = np.zeros(len(matrices), dtype=np.int64)
        batch_n = np.zeros(len(matrices), dtype=np.int64)
        for i, mat in enumerate(matrices):
            r, c, v = self._coo_triplets_cached(mat)
            batch_rows.append(r)
            batch_cols.append(c)
            batch_vals.append(v)
            batch_m[i] = int(mat.shape[0])
            batch_n[i] = int(mat.shape[1])
        results_jl = self.backend.batch_exact_snf_sparse(
            batch_rows, batch_cols, batch_vals, batch_m, batch_n
        )
        return [np.array(r, dtype=np.int64) for r in results_jl]

    def compute_sparse_rank_q(self, matrix) -> int:
        """Compute matrix rank over Q using Julia backend from sparse COO data.

        Args:
            matrix: The sparse matrix.

        Returns:
            The rank of the matrix over the rationals.
        """
        self.require_julia()
        if matrix is None or matrix.nnz == 0:
            return 0
        rows, cols, vals = self._coo_triplets_cached(matrix)
        rank = self.backend.rank_q_sparse(
            rows,
            cols,
            vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        )
        return int(rank)

    def compute_sparse_rank_mod_p(self, matrix, p: int) -> int:
        """Compute matrix rank over Z/pZ using Julia backend from sparse COO data.

        Args:
            matrix: The sparse matrix.
            p: The prime modulus.

        Returns:
            The rank of the matrix over Z/pZ.
        """
        self.require_julia()
        if matrix is None or matrix.nnz == 0:
            return 0
        rows, cols, vals = self._coo_triplets_cached(matrix)
        rank = self.backend.rank_mod_p_sparse(
            rows,
            cols,
            vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            int(p),
        )
        return int(rank)

    def compute_sparse_cohomology_basis(
        self, d_np1, d_n, cn_size: int | None = None
    ) -> list:
        """Executes the exact Julia sparse cohomology basis extraction Z^n / B^n.

        Args:
            d_np1: The boundary matrix d_{n+1}.
            d_n: The boundary matrix d_n.
            cn_size: The size of the n-chain group.

        Returns:
            A list of vectors forming a basis for the cohomology group.
        """
        self.require_julia()

        if d_np1 is None or d_np1.nnz == 0:
            d_np1_rows = np.array([], dtype=np.int64)
            d_np1_cols = np.array([], dtype=np.int64)
            d_np1_vals = np.array([], dtype=np.int64)
            d_np1_m, d_np1_n = (cn_size, 0) if cn_size is not None else (0, 0)
        else:
            d_np1_rows, d_np1_cols, d_np1_vals = self._coo_triplets_cached(d_np1)
            d_np1_m, d_np1_n = d_np1.shape

        if d_n is None or d_n.nnz == 0:
            d_n_rows = np.array([], dtype=np.int64)
            d_n_cols = np.array([], dtype=np.int64)
            d_n_vals = np.array([], dtype=np.int64)
            d_n_m, d_n_n = (0, cn_size) if cn_size is not None else (0, d_np1_m)
        else:
            d_n_rows, d_n_cols, d_n_vals = self._coo_triplets_cached(d_n)
            d_n_m, d_n_n = d_n.shape

        # Julia now returns a flat Matrix{Int64}
        basis_mat = self.backend.exact_sparse_cohomology_basis(
            d_np1_rows,
            d_np1_cols,
            d_np1_vals,
            int(d_np1_m),
            int(d_np1_n),
            d_n_rows,
            d_n_cols,
            d_n_vals,
            int(d_n_m),
            int(d_n_n),
        )

        # Convert columns to list of vectors
        basis_py = []
        for j in range(basis_mat.shape[1]):
            basis_py.append(basis_mat[:, j].astype(np.int64, copy=False))
        return basis_py

    def compute_sparse_cohomology_basis_mod_p(
        self, d_np1, d_n, p: int, cn_size: int | None = None
    ) -> list:
        """Executes Julia sparse cohomology basis extraction over Z/pZ for prime p.

        Args:
            d_np1: The boundary matrix d_{n+1}.
            d_n: The boundary matrix d_n.
            p: The prime modulus.
            cn_size: The size of the n-chain group.

        Returns:
            A list of vectors forming a basis for the cohomology group over Z/pZ.
        """
        self.require_julia()

        if d_np1 is None or d_np1.nnz == 0:
            d_np1_rows = np.array([], dtype=np.int64)
            d_np1_cols = np.array([], dtype=np.int64)
            d_np1_vals = np.array([], dtype=np.int64)
            d_np1_m, d_np1_n = (cn_size, 0) if cn_size is not None else (0, 0)
        else:
            d_np1_rows, d_np1_cols, d_np1_vals = self._coo_triplets_cached(d_np1)
            d_np1_m, d_np1_n = d_np1.shape

        if d_n is None or d_n.nnz == 0:
            d_n_rows = np.array([], dtype=np.int64)
            d_n_cols = np.array([], dtype=np.int64)
            d_n_vals = np.array([], dtype=np.int64)
            d_n_m, d_n_n = (0, cn_size) if cn_size is not None else (0, d_np1_m)
        else:
            d_n_rows, d_n_cols, d_n_vals = self._coo_triplets_cached(d_n)
            d_n_m, d_n_n = d_n.shape

        basis_mat = self.backend.sparse_cohomology_basis_mod_p(
            d_np1_rows,
            d_np1_cols,
            d_np1_vals,
            int(d_np1_m),
            int(d_np1_n),
            d_n_rows,
            d_n_cols,
            d_n_vals,
            int(d_n_m),
            int(d_n_n),
            int(p),
        )

        basis_py = []
        for j in range(basis_mat.shape[1]):
            basis_py.append(basis_mat[:, j].astype(np.int64, copy=False))
        return basis_py

    def compute_boundary_payload_from_simplices(
        self,
        simplex_entries: list,
        max_dim: int,
        *,
        include_metadata: bool = True,
    ) -> tuple:
        """Build boundary payloads through Julia with optional metadata suppression.

        Args:
            simplex_entries: A list of simplex tuples/lists.
            max_dim: The maximum dimension of simplices to consider.
            include_metadata: Whether to include metadata in the result.

        Returns:
            A tuple containing boundary matrices and related data.
        """
        self.require_julia()
        flat_vertices, simplex_offsets = self._flatten_simplices(simplex_entries)
        return self.backend.compute_boundary_payload_from_flat_simplices(
            flat_vertices,
            simplex_offsets,
            int(max_dim),
            bool(include_metadata),
        )

    def compute_boundary_payload_from_flat_simplices(
        self,
        flat_vertices: np.ndarray,
        simplex_offsets: np.ndarray,
        max_dim: int,
        *,
        include_metadata: bool = True,
    ) -> tuple:
        """Build boundary payloads through Julia using pre-flattened data.

        Args:
            flat_vertices: Flattened array of simplex vertices.
            simplex_offsets: Offsets into the flat_vertices array for each simplex.
            max_dim: The maximum dimension of simplices to consider.
            include_metadata: Whether to include metadata in the result.

        Returns:
            A tuple containing boundary matrices and related data.
        """
        self.require_julia()
        return self.backend.compute_boundary_payload_from_flat_simplices(
            np.asarray(flat_vertices, dtype=np.int64),
            np.asarray(simplex_offsets, dtype=np.int64),
            int(max_dim),
            bool(include_metadata),
        )

    def group_ring_multiply(
        self, coeffs1: dict, coeffs2: dict, group_order: int
    ) -> dict:
        """Multiply sparse group-ring coefficient dictionaries in the Julia backend.

        Args:
            coeffs1: The first coefficient dictionary.
            coeffs2: The second coefficient dictionary.
            group_order: The order of the group.

        Returns:
            The resulting coefficient dictionary after multiplication.
        """
        self.require_julia()
        # Direct passing of dictionaries; Julia side now uses pyconvert for speed.
        res_keys, res_vals = self.backend.group_ring_multiply(
            coeffs1, coeffs2, int(group_order)
        )

        def _key(k):
            ki = int(k)
            return "1" if ki == 0 else f"g_{ki}"

        return {
            _key(k): int(v) for k, v in zip(res_keys, res_vals)
        }

    def compute_multisignature(self, matrix: np.ndarray, p: int) -> int:
        """Evaluates L_{4k}(Z_p) obstruction by computing multisignature.

        Args:
            matrix: The symmetric real matrix.
            p: The prime modulus.

        Returns:
            The multisignature value.
        """
        self.require_julia()
        return int(
            self.backend.multisignature(np.asarray(matrix, dtype=np.float64), int(p))
        )

    def integral_lattice_isometry(
        self, matrix1: np.ndarray, matrix2: np.ndarray
    ) -> np.ndarray | None:
        """Find U in GL_n(Z) with U^T * matrix1 * U = matrix2 for definite forms.

        Args:
            matrix1: The first integral matrix.
            matrix2: The second integral matrix.

        Returns:
            The isometry matrix U if found, None otherwise.
        """
        self.require_julia()
        candidate = self.backend.integral_lattice_isometry(
            np.asarray(matrix1, dtype=np.int64), np.asarray(matrix2, dtype=np.int64)
        )
        if candidate is None:
            return None
        return np.array(candidate, dtype=np.int64)

    def abelianize_and_bhs_rank(self, generators: list, relations: list) -> tuple:
        """Computes the abelianization and extracts free and torsion ranks.

        Takes raw string generators and relations, computes the abelianization,
        and extracts the free and torsion ranks for the Bass-Heller-Swan K-theory formula.

        Args:
            generators: List of generator names.
            relations: List of relations.

        Returns:
            A tuple of (free_rank, torsion_ranks).
        """
        self.require_julia()
        flat_rels = [" ".join(r) for r in relations]
        # Short-circuit empty relations: free abelian group of rank len(generators).
        # Bypassing the Julia call here also avoids a multiple-dispatch recursion
        # bug in `abelianize_group` when `flat_rels` arrives as Vector{Any}.
        if not flat_rels:
            return int(len(generators)), []
        free_rank, torsions = self.backend.abelianize_group(generators, flat_rels)
        return int(free_rank), list(torsions)

    def compute_optimal_h1_basis_from_simplices(
        self,
        simplices: list,
        num_vertices: int,
        *,
        point_cloud: np.ndarray | None = None,
        max_roots: int | None = None,
        root_stride: int = 1,
        max_cycles: int | None = None,
    ) -> list:
        """Compute an optimal H1 basis via Julia backend (Algorithms 8+7 composition).

        Args:
            simplices: A list of simplex tuples/lists.
            num_vertices: Number of vertices in the complex.
            point_cloud: Optional (N, D) array of vertex coordinates.
            max_roots: Maximum number of root vertices for cycle search.
            root_stride: Stride for selecting root vertices.
            max_cycles: Maximum number of cycles to extract.

        Returns:
            A list of representative cycles as lists of edges.
        """
        self.require_julia()
        pts = (
            np.asarray(point_cloud, dtype=np.float64)
            if point_cloud is not None
            else self.jl.nothing
        )
        mr = int(max_roots) if max_roots is not None else self.jl.nothing
        mc = int(max_cycles) if max_cycles is not None else self.jl.nothing
        out = self.backend.optgen_from_simplices(
            simplices, int(num_vertices), pts, mr, int(root_stride), mc
        )
        basis_py = []
        for g in out:
            # g is a dict: {"dimension", "support_simplices", "support_edges", "weight", "certified_cycle"}
            basis_py.append([tuple((int(e[0]), int(e[1]))) for e in g["support_edges"]])
        return basis_py

    def compute_homology_basis_from_simplices(
        self,
        simplices: list,
        num_vertices: int,
        dimension: int,
        *,
        mode: str = "valid",
        point_cloud: np.ndarray | None = None,
        max_roots: int | None = None,
        root_stride: int = 1,
        max_cycles: int | None = None,
    ) -> list[dict]:
        """Compute H_k generator representatives from simplices via Julia backend over Z/2.

        Args:
            simplices: A list of simplex tuples/lists.
            num_vertices: Number of vertices in the complex.
            dimension: The dimension k of the homology group H_k.
            mode: Search mode for generators ('valid', 'optimized', etc.).
            point_cloud: Optional (N, D) array of vertex coordinates.
            max_roots: Maximum number of root vertices for cycle search.
            root_stride: Stride for selecting root vertices.
            max_cycles: Maximum number of cycles to extract.

        Returns:
            A list of dictionaries representing homology generators.
        """
        self.require_julia()
        pts = (
            np.asarray(point_cloud, dtype=np.float64)
            if point_cloud is not None
            else self.jl.nothing
        )
        mr = int(max_roots) if max_roots is not None else self.jl.nothing
        mc = int(max_cycles) if max_cycles is not None else self.jl.nothing
        out = self.backend.homology_generators_from_simplices(
            simplices,
            int(num_vertices),
            int(dimension),
            str(mode),
            pts,
            mr,
            int(root_stride),
            mc
        )

        parsed: list[dict] = []
        for g in out:
            support_simplices = [
                tuple(int(x) for x in simplex) for simplex in g["support_simplices"]
            ]
            support_edges = [tuple((int(e[0]), int(e[1]))) for e in g["support_edges"]]
            parsed.append(
                {
                    "dimension": int(g["dimension"]),
                    "support_simplices": support_simplices,
                    "support_edges": support_edges,
                    "weight": float(g["weight"]),
                    "certified_cycle": bool(g["certified_cycle"]),
                }
            )
        return parsed

    def compute_pi1_raw_data(
        self,
        d1_rows: np.ndarray,
        d1_cols: np.ndarray,
        d1_vals: np.ndarray,
        n_vertices: int,
        n_edges: int,
        d2_rows: np.ndarray,
        d2_cols: np.ndarray,
        d2_vals: np.ndarray,
        n_faces: int,
    ) -> dict:
        """Compute full raw pi1 data (generators, relations, traces) via Julia.

        Args:
            d1_rows: Row indices of the d1 boundary matrix.
            d1_cols: Column indices of the d1 boundary matrix.
            d1_vals: Values of the d1 boundary matrix.
            n_vertices: Number of vertices.
            n_edges: Number of edges.
            d2_rows: Row indices of the d2 boundary matrix.
            d2_cols: Column indices of the d2 boundary matrix.
            d2_vals: Values of the d2 boundary matrix.
            n_faces: Number of faces (2-cells).

        Returns:
            A dictionary with 'generators', 'relations', and 'traces'.
        """
        self.require_julia()
        res = self.backend.extract_pi1_raw_data_jl(
            np.asarray(d1_rows, dtype=np.int64),
            np.asarray(d1_cols, dtype=np.int64),
            np.asarray(d1_vals, dtype=np.int64),
            int(n_vertices),
            int(n_edges),
            np.asarray(d2_rows, dtype=np.int64),
            np.asarray(d2_cols, dtype=np.int64),
            np.asarray(d2_vals, dtype=np.int64),
            int(n_faces),
        )
        
        # Parse traces
        parsed_traces = []
        for tr in res["traces"]:
            parsed_traces.append({
                "generator": str(tr["generator"]),
                "edge_index": int(tr["edge_index"]),
                "component_root": int(tr["component_root"]),
                "vertex_path": [int(x) for x in tr["vertex_path"]],
                "directed_edge_path": [(int(e[0]), int(e[1])) for e in tr["directed_edge_path"]],
                "undirected_edge_path": [(int(e[0]), int(e[1])) for e in tr["undirected_edge_path"]],
            })
            
        return {
            "generators": {int(k): str(v) for k, v in dict(res["generators"]).items()},
            "relations": [[str(t) for t in r] for r in res["relations"]],
            "traces": parsed_traces
        }

    def compute_pi1_trace_candidates(
        self,
        d1_rows: np.ndarray,
        d1_cols: np.ndarray,
        d1_vals: np.ndarray,
        *,
        n_vertices: int,
        n_edges: int,
    ) -> list[dict]:
        """Compute raw pi1 generator trace candidates from d1 COO data via Julia.

        Args:
            d1_rows: Row indices of the d1 boundary matrix.
            d1_cols: Column indices of the d1 boundary matrix.
            d1_vals: Values of the d1 boundary matrix.
            n_vertices: Number of vertices.
            n_edges: Number of edges.

        Returns:
            A list of dictionaries representing pi1 generator trace candidates.
        """
        self.require_julia()
        out = self.backend.pi1_trace_candidates_from_d1(
            np.asarray(d1_rows, dtype=np.int64),
            np.asarray(d1_cols, dtype=np.int64),
            np.asarray(d1_vals, dtype=np.int64),
            int(n_vertices),
            int(n_edges),
        )
        parsed: list[dict] = []
        for tr in out:
            parsed.append(
                {
                    "generator": str(tr["generator"]),
                    "edge_index": int(tr["edge_index"]),
                    "component_root": int(tr["component_root"]),
                    "vertex_path": [int(x) for x in tr["vertex_path"]],
                    "directed_edge_path": [
                        (int(e[0]), int(e[1])) for e in tr["directed_edge_path"]
                    ],
                    "undirected_edge_path": [
                        (int(e[0]), int(e[1])) for e in tr["undirected_edge_path"]
                    ],
                }
            )
        return parsed

    def compute_boundary_data_from_simplices(
        self, simplex_entries: list, max_dim: int
    ) -> tuple[dict, dict, dict, dict]:
        """Build boundary COO payloads and simplex tables through Julia for large simplicial workloads.

        Args:
            simplex_entries: A list of simplex tuples/lists.
            max_dim: The maximum dimension of simplices to consider.

        Returns:
            A tuple of (boundaries, cells, dim_simplices, simplex_to_idx) dictionaries.
        """
        self.require_julia()
        result = self.backend.compute_boundary_data_from_simplices_jl(
            simplex_entries,
            int(max_dim),
        )

        boundaries_jl, cells_jl, dim_simplices_jl, simplex_to_idx_jl = result

        boundaries_py: dict[int, dict[str, object]] = {}
        for k, payload in dict(boundaries_jl).items():
            kk = int(k)
            boundaries_py[kk] = {
                "rows": np.asarray(payload["rows"], dtype=np.int64),
                "cols": np.asarray(payload["cols"], dtype=np.int64),
                "data": np.asarray(payload["data"], dtype=np.int64),
                "n_rows": int(payload["n_rows"]),
                "n_cols": int(payload["n_cols"]),
            }

        cells_py = {int(k): int(v) for k, v in dict(cells_jl).items()}

        # dim_simplices_jl can return matrices (d+1, N) or lists of tuples
        dim_simplices_py = {}
        for k, val in dict(dim_simplices_jl).items():
            kk = int(k)
            if hasattr(val, "shape") and len(val.shape) == 2:
                # val is (d+1, N)
                dim_simplices_py[kk] = [
                    tuple(int(x) for x in val[:, j]) for j in range(val.shape[1])
                ]
            else:
                # val is a sequence of tuples/vectors
                dim_simplices_py[kk] = [
                    tuple(int(x) for x in s) for s in val
                ]

        simplex_to_idx_py = {
            int(k): {
                tuple(int(x) for x in simplex): int(idx)
                for simplex, idx in dict(idx_map).items()
            }
            for k, idx_map in dict(simplex_to_idx_jl).items()
            if len(dict(idx_map)) > 0
        }
        return boundaries_py, cells_py, dim_simplices_py, simplex_to_idx_py

    def compute_boundary_mod2_matrix(
        self, source_simplices: list, target_simplices: list
    ) -> dict:
        """Compute mod-2 boundary matrix through Julia for fast homology generator extraction.

        Args:
            source_simplices: List of source simplices.
            target_simplices: List of target simplices.

        Returns:
            A dictionary containing the sparse matrix payload.
        """
        self.require_julia()
        payload = self.backend.compute_boundary_mod2_matrix(
            source_simplices, target_simplices
        )
        return {
            "rows": np.asarray(payload["rows"], dtype=np.int64),
            "cols": np.asarray(payload["cols"], dtype=np.int64),
            "data": np.asarray(payload["data"], dtype=np.int64),
            "m": int(payload["m"]),
            "n": int(payload["n"]),
        }

    def compute_alexander_whitney_cup(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        p: int,
        q: int,
        simplices_p_plus_q: list,
        simplex_to_idx_p: dict,
        simplex_to_idx_q: dict,
        modulus: int | None = None,
    ) -> np.ndarray:
        """Compute Alexander-Whitney cup product through Julia for fast intersection form extraction.

        Args:
            alpha: Cochain coefficients for the first class.
            beta: Cochain coefficients for the second class.
            p: Dimension of the first class.
            q: Dimension of the second class.
            simplices_p_plus_q: List of (p+q)-dimensional simplices.
            simplex_to_idx_p: Mapping from p-simplices to their indices.
            simplex_to_idx_q: Mapping from q-simplices to their indices.
            modulus: Optional modulus for coefficients.

        Returns:
            An array containing the cup product coefficients.
        """
        self.require_julia()
        result = self.backend.compute_alexander_whitney_cup(
            np.asarray(alpha, dtype=np.int64),
            np.asarray(beta, dtype=np.int64),
            int(p),
            int(q),
            simplices_p_plus_q,
            simplex_to_idx_p,
            simplex_to_idx_q,
            int(modulus) if modulus is not None else self.jl.nothing,
        )
        return np.asarray(result, dtype=np.int64)

    def compute_trimesh_boundary_data(self, faces: list, n_vertices: int) -> dict:
        """Compute trimesh boundary operators (d1, d2) through Julia.

        Args:
            faces: List of triangular faces.
            n_vertices: Number of vertices.

        Returns:
            A dictionary containing d1 and d2 sparse matrix payloads.
        """
        self.require_julia()
        if isinstance(faces, np.ndarray) and faces.ndim == 2:
            faces_arr = np.asarray(faces, dtype=np.int64)
            flat = np.ascontiguousarray(faces_arr.reshape(-1), dtype=np.int64)
            offsets = np.arange(0, flat.size + 1, faces_arr.shape[1], dtype=np.int64)
            payload = self.backend.compute_trimesh_boundary_data_flat(
                flat,
                offsets,
                int(n_vertices),
            )
        else:
            flat, offsets = self._flatten_simplices(list(faces))
            payload = self.backend.compute_trimesh_boundary_data_flat(
                flat,
                offsets,
                int(n_vertices),
            )
        return {
            "d1_rows": np.asarray(payload["d1_rows"], dtype=np.int64),
            "d1_cols": np.asarray(payload["d1_cols"], dtype=np.int64),
            "d1_data": np.asarray(payload["d1_data"], dtype=np.int64),
            "n_vertices": int(payload["n_vertices"]),
            "n_edges": int(payload["n_edges"]),
            "d2_rows": np.asarray(payload["d2_rows"], dtype=np.int64),
            "d2_cols": np.asarray(payload["d2_cols"], dtype=np.int64),
            "d2_data": np.asarray(payload["d2_data"], dtype=np.int64),
            "n_faces": int(payload["n_faces"]),
        }

    def triangulate_surface_delaunay(
        self, points: np.ndarray, tolerance: float = 1e-10
    ) -> list:
        """Triangulate a 2D surface from a point cloud using Delaunay triangulation.

        Args:
            points: An (N, 2) array of point coordinates.
            tolerance: Geometric tolerance for triangulation.

        Returns:
            A list of triangle faces as vertex index lists.
        """
        self.require_julia()
        triangles = self.backend.triangulate_surface_delaunay(
            np.asarray(points, dtype=np.float64), float(tolerance)
        )
        return [list(tri) for tri in triangles]

    def compute_tangential_local_stars_2d(
        self, flat_global_idx: np.ndarray, flat_coords: np.ndarray, offsets: np.ndarray
    ) -> tuple:
        """Per-point local-star construction (k=2 only) via DelaunayTriangulation.jl.

        Args:
            flat_global_idx: (M,) flattened global point indices of every
                point's local neighborhood (CSR-style, see ``offsets``); each
                neighborhood's own center point is at its own first position.
            flat_coords: (M, 2) flattened tangent-projected local
                coordinates, row-aligned with ``flat_global_idx``.
            offsets: (N + 1,) CSR offsets into the two flat arrays above.

        Returns:
            A tuple ``(star_flat, star_offsets, ok)``: ``ok[i]`` is False iff
            point ``i``'s own local triangulation failed outright;
            ``star_flat``/``star_offsets`` are a CSR-style (offsets in units
            of triangles) list of each point's own local star, as 0-based
            global-index triples.

        Raises:
            RuntimeError: If the Julia call fails (including when
                DelaunayTriangulation.jl itself is not installed).
        """
        self.require_julia()
        try:
            star_flat, star_offsets, ok = self.backend.compute_tangential_local_stars_2d_jl(
                np.ascontiguousarray(flat_global_idx, dtype=np.int64),
                np.ascontiguousarray(flat_coords, dtype=np.float64),
                np.ascontiguousarray(offsets, dtype=np.int64),
            )
            return (
                np.asarray(star_flat, dtype=np.int64),
                np.asarray(star_offsets, dtype=np.int64),
                np.asarray(ok, dtype=bool),
            )
        except Exception as e:
            raise RuntimeError(f"compute_tangential_local_stars_2d failed: {e!r}")

    def orthogonal_procrustes(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Solves the orthogonal Procrustes problem using Julia.

        Args:
            A: The first point cloud matrix.
            B: The second point cloud matrix.

        Returns:
            A tuple of (Rotation matrix R, aligned matrix B_aligned, disparity).
        """
        self.require_julia()
        res = self.backend.orthogonal_procrustes(
            np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
        )
        # res is a Tuple: (R, B_aligned, disparity)
        return np.asarray(res[0]), np.asarray(res[1]), float(res[2])

    def pairwise_distance_matrix(self, data: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        """Computes a pairwise distance matrix using Julia.

        Args:
            data: An (N, D) array of points.
            metric: The distance metric to use.

        Returns:
            An (N, N) pairwise distance matrix.
        """
        self.require_julia()
        res = self.backend.pairwise_distance_matrix(
            np.asarray(data, dtype=np.float64), str(metric)
        )
        return np.asarray(res)

    def frechet_distance(self, curve_a: np.ndarray, curve_b: np.ndarray) -> float:
        """Computes the discrete Frechet distance between two curves using Julia.

        Args:
            curve_a: An (N, D) array representing the first curve.
            curve_b: An (M, D) array representing the second curve.

        Returns:
            The discrete Frechet distance.
        """
        self.require_julia()
        res = self.backend.frechet_distance(
            np.asarray(curve_a, dtype=np.float64), np.asarray(curve_b, dtype=np.float64)
        )
        return float(res)

    def gromov_wasserstein_distance(
        self, D_A: np.ndarray, D_B: np.ndarray, p: np.ndarray, q: np.ndarray, epsilon: float, max_iter: int
    ) -> float:
        """Computes the entropic Gromov-Wasserstein distance using Julia.

        Args:
            D_A: Distance matrix of the first space.
            D_B: Distance matrix of the second space.
            p: Probability distribution over points in the first space.
            q: Probability distribution over points in the second space.
            epsilon: Regularization parameter.
            max_iter: Maximum number of Sinkhorn iterations.

        Returns:
            The entropic Gromov-Wasserstein distance.
        """
        self.require_julia()
        res = self.backend.gromov_wasserstein_distance(
            np.asarray(D_A, dtype=np.float64),
            np.asarray(D_B, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
            np.asarray(q, dtype=np.float64),
            float(epsilon),
            int(max_iter)
        )
        return float(res)


    def enumerate_cliques_sparse(self, rowptr: np.ndarray, colval: np.ndarray, n_vertices: int, max_dim: int) -> list:
        """Enumerates cliques in a sparse graph using Julia.

        Args:
            rowptr: Row pointers of the sparse adjacency matrix.
            colval: Column values of the sparse adjacency matrix.
            n_vertices: Number of vertices in the graph.
            max_dim: Maximum dimension of cliques to enumerate.

        Returns:
            A list of cliques, where each clique is a list of vertex indices.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            res = self.backend.enumerate_cliques_sparse(
                rowptr,
                colval,
                int(n_vertices),
                int(max_dim)
            )
            return [[int(x) for x in c] for c in res]
        except Exception as e:
            raise RuntimeError(f"enumerate_cliques_sparse failed: {e!r}")

    def compute_circumradius_sq_3d(self, points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        """Computes squared circumradii for 3D simplices using Julia.

        Args:
            points: An (N, 3) array of point coordinates.
            simplices: An (M, 4) array of tetrahedra vertex indices.

        Returns:
            An (M,) array of squared circumradii.

        Raises:
            RuntimeError: If the Julia backend is unavailable or the call fails.
        """
        if not self.available:
            raise RuntimeError("Julia backend unavailable.")
        try:
            res = self.backend.compute_circumradius_sq_3d(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64),  # 0-based
            )
            return np.array(res, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"compute_circumradius_sq_3d failed: {e!r}")

    def compute_circumradius_sq_2d(self, points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        """Computes squared circumradii for 2D simplices using Julia.

        Args:
            points: An (N, 2) array of point coordinates.
            simplices: An (M, 3) array of triangle vertex indices.

        Returns:
            An (M,) array of squared circumradii.

        Raises:
            RuntimeError: If the Julia backend is unavailable or the call fails.
        """
        if not self.available:
            raise RuntimeError("Julia backend unavailable.")
        try:
            res = self.backend.compute_circumradius_sq_2d(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64),  # 0-based
            )
            return np.array(res, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"compute_circumradius_sq_2d failed: {e!r}")

    def exact_signs_of_determinants_batch(self, matrices: np.ndarray) -> np.ndarray:
        """Computes exact determinant signs for a batch of small square matrices using Julia.

        Mirrors ``pysurgery.geometry.predicates.exact_signs_of_determinants_batch``'s
        two-tier algorithm (float64 Leibniz formula, Higham ``gamma_k`` error bound, exact
        fallback -- here ``Rational{BigInt}`` rather than ``fractions.Fraction``) so the two
        backends are expected to agree exactly, not merely approximately.

        Args:
            matrices: An (M, n, n) array of small square matrices, 1 <= n <= 6.

        Returns:
            An (M,) int64 array of signs in {-1, 0, 1}.

        Raises:
            RuntimeError: If the Julia backend is unavailable or the call fails.
        """
        if not self.available:
            raise RuntimeError("Julia backend unavailable.")
        try:
            res = self.backend.exact_signs_of_determinants_batch_jl(
                np.asarray(matrices, dtype=np.float64),
            )
            return np.array(res, dtype=np.int64)
        except Exception as e:
            raise RuntimeError(f"exact_signs_of_determinants_batch failed: {e!r}")

    def compute_alpha_complex_simplices(
        self,
        points: np.ndarray,
        max_simplices: np.ndarray,
        alpha2: float,
        max_dim: int,
    ) -> list[tuple[int, ...]]:
        """Compute filtered Alpha Complex simplices via Julia backend.

        Args:
            points: An (N, D) array of point coordinates.
            max_simplices: Array of maximal simplices (e.g., from Delaunay).
            alpha2: Squared alpha parameter.
            max_dim: Maximum dimension of simplices to return.

        Returns:
            A list of simplex tuples.
        """
        self.require_julia()
        res = self.backend.compute_alpha_complex_simplices_jl(
            np.asarray(points, dtype=np.float64),
            np.asarray(max_simplices, dtype=np.int64),
            float(alpha2),
            int(max_dim),
        )
        # res is Vector{Vector{Int64}} from Julia
        return [tuple(int(x) for x in s) for s in res]

    def compute_cknn_graph(self, pts: np.ndarray, k: int, delta: float) -> np.ndarray:
        """Compute the Continuous k-Nearest Neighbors graph using Julia.

        Args:
            pts: An (N, D) array of point coordinates.
            k: Number of neighbors.
            delta: The delta parameter for CkNN.

        Returns:
            An (M, 2) array of edge index pairs.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        pts_view = np.asarray(pts, dtype=np.float64, order="C")
        try:
            with self._lock:
                out = self.backend.cknn_graph_jl(pts_view, int(k), float(delta))
                pairs = np.array(out, dtype=np.int64)
                if pairs.size == 0:
                    return np.zeros((0, 2), dtype=np.int64)
                return pairs.reshape(-1, 2)
        except Exception as e:
            raise RuntimeError(f"Julia cknn_graph failed: {e!r}")

    
    def compute_cknn_graph_accelerated(self, pts: np.ndarray, rho: np.ndarray, delta: float) -> np.ndarray:
        """Compute the CkNN graph using pre-computed rho values for speed.

        Args:
            pts: An (N, D) array of point coordinates.
            rho: An (N,) array of pre-computed neighbor distances.
            delta: The delta parameter for CkNN.

        Returns:
            An (M, 2) array of edge index pairs.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        pts_view = np.asarray(pts, dtype=np.float64, order="C")
        rho_view = np.asarray(rho, dtype=np.float64)
        try:
            with self._lock:
                out = self.backend.cknn_graph_accelerated_jl(pts_view, rho_view, float(delta))
                pairs = np.array(out, dtype=np.int64)
                if pairs.size == 0:
                    return np.zeros((0, 2), dtype=np.int64)
                return pairs.reshape(-1, 2)
        except Exception as e:
            raise RuntimeError(f"Julia cknn_graph_accelerated failed: {e!r}")

    def quick_mapper_jl(self, G_raw: dict, max_loops: int = 1, min_modularity_gain: float = 1e-6) -> tuple[dict, dict]:
        """Executes the high-performance QuickMapper algorithm in Julia.

        Args:
            G_raw: A dict with keys "V" (list of ints) and "E" (list of tuples of ints).
            max_loops: Maximum number of simplification loops.
            min_modularity_gain: Minimum modularity gain to continue.

        Returns:
            A tuple of (simplified_graph_dict, mapping_dict_L).

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            G_simple, L_jl = self.backend.quick_mapper_jl(
                G_raw,
                int(max_loops),
                float(min_modularity_gain)
            )
            L_py = dict(L_jl)
            G_simple_py = dict(G_simple)
            if "E" in G_simple_py:
                G_simple_py["E"] = [tuple(e) for e in G_simple_py["E"]]
            return G_simple_py, L_py
        except Exception as e:
            raise RuntimeError(f"quick_mapper_jl failed: {e!r}")

    def compute_persistence_barcodes(self, boundary_matrices: dict[int, sp.spmatrix], field: str) -> list[dict]:
        """Compute persistent homology barcodes via Julia backend.

        Args:
            boundary_matrices: Dictionary mapping dimension d to boundary matrix D_d.
            field: Field for computation ('Z2' or 'Q').

        Returns:
            A list of dictionaries containing barcode data (birth, death, dim, multiplicity).
        """
        self.require_julia()
        
        jl_dict = self.jl.Dict[self.jl.Int, self.jl.SparseArrays.SparseMatrixCSC]()
        
        for d, mat in boundary_matrices.items():
            if not sp.issparse(mat):
                mat = sp.csr_matrix(mat)
            
            # Use cached COO triplets for speed
            rows, cols, vals = self._coo_triplets_cached(mat)
            
            # Convert to SparseMatrixCSC in Julia
            # Julia indices are 1-based, triplets from _coo_triplets_cached are 0-based
            jl_mat = self.jl.SparseArrays.sparse(
                self.jl.convert(self.jl.Vector[self.jl.Int], rows + 1),
                self.jl.convert(self.jl.Vector[self.jl.Int], cols + 1),
                self.jl.convert(self.jl.Vector[self.jl.Int64], vals),
                int(mat.shape[0]),
                int(mat.shape[1])
            )
            jl_dict[int(d)] = jl_mat
            
        jl_field = self.jl.Symbol(field)
        
        try:
            jl_barcodes = self.backend.compute_persistence_barcodes(jl_dict, jl_field)
        except Exception as e:
            raise RuntimeError(f"Julia compute_persistence_barcodes failed: {e}")
            
        barcodes = []
        for b in jl_barcodes:
            barcodes.append({
                "birth": int(b.birth) - 1, # Convert back to 0-indexed
                "death": int(b.death) - 1,
                "dim": int(b.dim),
                "multiplicity": int(b.multiplicity)
            })

        return barcodes

    def compute_filtration_persistence(
        self,
        simplices_flat: np.ndarray,
        simplex_ptr: np.ndarray,
        vals: np.ndarray,
    ) -> list[tuple[int, float, float]]:
        """Exact Z2 persistence of a monotone filtration via twist/clearing in Julia.

        This is the high-performance replacement for the pure-Python column
        reduction in ``FiltrationReport``: a single filtration-ordered boundary
        reduction with the Chen-Kerber clearing optimisation. The result is exact
        (identical to the naive reduction), not an approximation.

        Args:
            simplices_flat: 1-D int array, the vertices of every simplex of the
                maximal complex concatenated in any order.
            simplex_ptr: int array of length ``n_simplices + 1``; simplex ``j``
                spans ``simplices_flat[simplex_ptr[j]:simplex_ptr[j+1]]``.
            vals: float array of per-simplex appearance values (aligned with
                ``simplex_ptr``).

        Returns:
            A list of ``(dim, birth, death)`` tuples; ``death`` is ``float('inf')``
            for essential classes and zero-persistence pairs are omitted.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            bar_dim, bar_birth, bar_death = self.backend.compute_filtration_persistence(
                np.ascontiguousarray(simplices_flat, dtype=np.int64),
                np.ascontiguousarray(simplex_ptr, dtype=np.int64),
                np.ascontiguousarray(vals, dtype=np.float64),
            )
            # .tolist() is a single C-level conversion; far cheaper than a Python
            # per-element loop over a potentially multi-million-entry barcode.
            dims = np.asarray(bar_dim, dtype=np.int64).tolist()
            births = np.asarray(bar_birth, dtype=np.float64).tolist()
            deaths = np.asarray(bar_death, dtype=np.float64).tolist()
            return list(zip(dims, births, deaths))
        except Exception as e:
            raise RuntimeError(f"compute_filtration_persistence failed: {e!r}")

    def compute_rips_filtration(
        self,
        points: np.ndarray,
        epsilon: float,
        max_dim: int,
        analyze_manifolds: bool = False,
        n_samples: Optional[int] = None,
        verify_manifold_only_at_betti_change: bool = False,
        track_connected_components: bool = False,
    ) -> dict:
        """Fused Vietoris-Rips build + longest-edge filtration + Z2 persistence in Julia.

        Does the *entire* hot path inside Julia -- build the VR complex, assign each
        simplex its longest-edge appearance value, and run the exact twist/clearing
        reduction -- so the full simplex set never crosses back to Python (only the
        barcode and small summaries do). This avoids the Python->Julia->Python clique
        and boundary-matrix marshaling that otherwise dominates large reports. The
        barcode is exact (identical to the staged reduction), not an approximation.

        Args:
            points: (N, D) array of point coordinates.
            epsilon: Maximum edge length (diameter cap) for the complex.
            max_dim: Maximum simplex dimension to build.
            analyze_manifolds: If True, run the per-threshold manifold analysis in Julia.
            n_samples: If given, select this many thresholds from distinct values.
            verify_manifold_only_at_betti_change: If True, check manifold criteria only when Betti numbers change.
            track_connected_components: If True, track connected components.

        Returns:
            A dict with keys:
                ``barcode`` -- list of ``(dim, birth, death)`` tuples;
                ``eps_values`` -- sorted distinct appearance values;
                ``dim_first_appear`` -- ``{dim: minimum appearance value}``;
                ``total`` -- total number of simplices in the complex;
                ``manifold_data`` -- dict of manifold results per threshold if analyze_manifolds is True.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            res = self.backend.compute_rips_filtration(
                np.ascontiguousarray(points, dtype=np.float64),
                float(epsilon),
                int(max_dim),
                bool(analyze_manifolds),
                n_samples,
                bool(verify_manifold_only_at_betti_change),
                bool(track_connected_components),
            )
            bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val, _dim_count, total = res[0:8]
            dims = np.asarray(bar_dim, dtype=np.int64).tolist()
            births = np.asarray(bar_birth, dtype=np.float64).tolist()
            deaths = np.asarray(bar_death, dtype=np.float64).tolist()
            barcode = list(zip(dims, births, deaths))

            ids = np.asarray(dim_ids, dtype=np.int64).tolist()
            firsts = np.asarray(dim_first_val, dtype=np.float64).tolist()
            payload = {
                "barcode": barcode,
                "eps_values": np.asarray(eps_values, dtype=np.float64).tolist(),
                "dim_first_appear": {int(d): float(f) for d, f in zip(ids, firsts)},
                "total": int(total),
            }

            if len(res) > 8 and res[8]:
                payload["manifold_data"] = {
                    "epsilons": np.asarray(res[9], dtype=np.float64).tolist(),
                    "is_manifold": np.asarray(res[10], dtype=bool).tolist(),
                    "dimensions": np.asarray(res[11], dtype=np.int64).tolist(),
                    "is_closed": np.asarray(res[12], dtype=bool).tolist(),
                    "failures": np.asarray(res[13], dtype=np.int64).tolist(),
                }
                if len(res) > 14 and len(res[14]) > 0:
                    comp_keys = [np.asarray(x, dtype=np.int64).tolist() for x in res[14]]
                    comp_vals = [list(x) for x in res[15]]
                    payload["component_data"] = [dict(zip(k, v)) for k, v in zip(comp_keys, comp_vals)]
                else:
                    payload["component_data"] = None
            else:
                payload["manifold_data"] = None
                payload["component_data"] = None

            return payload
        except Exception as e:
            raise RuntimeError(f"compute_rips_filtration failed: {e!r}")

    def compute_alpha_filtration(
        self,
        points: np.ndarray,
        top_simplices: np.ndarray,
        max_dim: int,
        analyze_manifolds: bool = False,
        n_samples: Optional[int] = None,
        eps_max: Optional[float] = None,
        verify_manifold_only_at_betti_change: bool = False,
        track_connected_components: bool = False,
    ) -> dict:
        """Fused Alpha complex build + Gabriel testing + Z2 persistence in Julia.

        Args:
            points: (N, D) array of point coordinates.
            top_simplices: (M, D + 1) array of Delaunay top simplices (vertex indices).
            max_dim: Maximum simplex dimension to build.
            analyze_manifolds: If True, run the per-threshold manifold analysis in Julia.
            n_samples: If given, select this many thresholds from distinct values.
            eps_max: If given, maximum filtration value to include.
            verify_manifold_only_at_betti_change: If True, check manifold criteria only when Betti numbers change.
            track_connected_components: If True, track connected components.

        Returns:
            A dict with keys:
                ``barcode`` -- list of ``(dim, birth, death)`` tuples;
                ``eps_values`` -- sorted distinct appearance values;
                ``dim_first_appear`` -- ``{dim: minimum appearance value}``;
                ``total`` -- total number of simplices in the complex;
                ``manifold_data`` -- dict of manifold results per threshold if analyze_manifolds is True.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            res = self.backend.compute_alpha_filtration(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(top_simplices, dtype=np.int64),
                int(max_dim),
                bool(analyze_manifolds),
                n_samples,
                float(eps_max) if eps_max is not None else None,
                bool(verify_manifold_only_at_betti_change),
                bool(track_connected_components),
            )
            bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val, _dim_count, total = res[0:8]
            dims = np.asarray(bar_dim, dtype=np.int64).tolist()
            births = np.asarray(bar_birth, dtype=np.float64).tolist()
            deaths = np.asarray(bar_death, dtype=np.float64).tolist()
            barcode = list(zip(dims, births, deaths))

            ids = np.asarray(dim_ids, dtype=np.int64).tolist()
            firsts = np.asarray(dim_first_val, dtype=np.float64).tolist()
            payload = {
                "barcode": barcode,
                "eps_values": np.asarray(eps_values, dtype=np.float64).tolist(),
                "dim_first_appear": {int(d): float(f) for d, f in zip(ids, firsts)},
                "total": int(total),
            }

            if len(res) > 8 and res[8]:
                payload["manifold_data"] = {
                    "epsilons": np.asarray(res[9], dtype=np.float64).tolist(),
                    "is_manifold": np.asarray(res[10], dtype=bool).tolist(),
                    "dimensions": np.asarray(res[11], dtype=np.int64).tolist(),
                    "is_closed": np.asarray(res[12], dtype=bool).tolist(),
                    "failures": np.asarray(res[13], dtype=np.int64).tolist(),
                }
                if len(res) > 14 and len(res[14]) > 0:
                    comp_keys = [np.asarray(x, dtype=np.int64).tolist() for x in res[14]]
                    comp_vals = [list(x) for x in res[15]]
                    payload["component_data"] = [dict(zip(k, v)) for k, v in zip(comp_keys, comp_vals)]
                else:
                    payload["component_data"] = None
            else:
                payload["manifold_data"] = None
                payload["component_data"] = None

            return payload
        except Exception as e:
            raise RuntimeError(f"compute_alpha_filtration failed: {e!r}")

    def _compute_delaunay_restricted_filtration(
        self,
        kernel_name: str,
        points: np.ndarray,
        top_simplices: np.ndarray,
        max_dim: int,
        analyze_manifolds: bool,
        n_samples: Optional[int],
        eps_max: Optional[float],
        verify_manifold_only_at_betti_change: bool,
        track_connected_components: bool,
    ) -> dict:
        """Shared payload assembly for the Delaunay-Rips/Cech fused kernels.

        Both ``compute_delaunay_rips_filtration`` and
        ``compute_delaunay_cech_filtration`` call the same Julia entry-point
        shape (see ``compute_alpha_filtration``) and build an identical payload
        dict; only the Julia function name differs.
        """
        self.require_julia()
        try:
            backend_fn = getattr(self.backend, kernel_name)
            res = backend_fn(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(top_simplices, dtype=np.int64),
                int(max_dim),
                bool(analyze_manifolds),
                n_samples,
                float(eps_max) if eps_max is not None else None,
                bool(verify_manifold_only_at_betti_change),
                bool(track_connected_components),
            )
            bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val, _dim_count, total = res[0:8]
            dims = np.asarray(bar_dim, dtype=np.int64).tolist()
            births = np.asarray(bar_birth, dtype=np.float64).tolist()
            deaths = np.asarray(bar_death, dtype=np.float64).tolist()
            barcode = list(zip(dims, births, deaths))

            ids = np.asarray(dim_ids, dtype=np.int64).tolist()
            firsts = np.asarray(dim_first_val, dtype=np.float64).tolist()
            payload = {
                "barcode": barcode,
                "eps_values": np.asarray(eps_values, dtype=np.float64).tolist(),
                "dim_first_appear": {int(d): float(f) for d, f in zip(ids, firsts)},
                "total": int(total),
            }

            if len(res) > 8 and res[8]:
                payload["manifold_data"] = {
                    "epsilons": np.asarray(res[9], dtype=np.float64).tolist(),
                    "is_manifold": np.asarray(res[10], dtype=bool).tolist(),
                    "dimensions": np.asarray(res[11], dtype=np.int64).tolist(),
                    "is_closed": np.asarray(res[12], dtype=bool).tolist(),
                    "failures": np.asarray(res[13], dtype=np.int64).tolist(),
                }
                if len(res) > 14 and len(res[14]) > 0:
                    comp_keys = [np.asarray(x, dtype=np.int64).tolist() for x in res[14]]
                    comp_vals = [list(x) for x in res[15]]
                    payload["component_data"] = [dict(zip(k, v)) for k, v in zip(comp_keys, comp_vals)]
                else:
                    payload["component_data"] = None
            else:
                payload["manifold_data"] = None
                payload["component_data"] = None

            return payload
        except Exception as e:
            raise RuntimeError(f"{kernel_name} failed: {e!r}")

    def compute_delaunay_rips_filtration(
        self,
        points: np.ndarray,
        top_simplices: np.ndarray,
        max_dim: int,
        analyze_manifolds: bool = False,
        n_samples: Optional[int] = None,
        eps_max: Optional[float] = None,
        verify_manifold_only_at_betti_change: bool = False,
        track_connected_components: bool = False,
    ) -> dict:
        """Fused Delaunay-restricted Rips build + longest-edge values + Z2 persistence in Julia.

        Structurally identical to ``compute_alpha_filtration`` (same Delaunay
        face-enumeration step), but each face's appearance value is the longest
        edge among its own vertices rather than a circumradius/Gabriel test --
        so, unlike Alpha, no coface propagation is needed.

        Args:
            points: (N, D) array of point coordinates.
            top_simplices: (M, D + 1) array of Delaunay top simplices (vertex indices).
            max_dim: Maximum simplex dimension to build.
            analyze_manifolds: If True, run the per-threshold manifold analysis in Julia.
            n_samples: If given, select this many thresholds from distinct values.
            eps_max: If given, maximum filtration value to include.
            verify_manifold_only_at_betti_change: If True, check manifold criteria only when Betti numbers change.
            track_connected_components: If True, track connected components.

        Returns:
            A dict with the same keys as ``compute_alpha_filtration``'s payload.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        return self._compute_delaunay_restricted_filtration(
            "compute_delaunay_rips_filtration", points, top_simplices, max_dim,
            analyze_manifolds, n_samples, eps_max,
            verify_manifold_only_at_betti_change, track_connected_components,
        )

    def compute_delaunay_cech_filtration(
        self,
        points: np.ndarray,
        top_simplices: np.ndarray,
        max_dim: int,
        analyze_manifolds: bool = False,
        n_samples: Optional[int] = None,
        eps_max: Optional[float] = None,
        verify_manifold_only_at_betti_change: bool = False,
        track_connected_components: bool = False,
    ) -> dict:
        """Fused Delaunay-restricted Cech build + min-enclosing-ball values + Z2 persistence in Julia.

        Structurally identical to ``compute_alpha_filtration`` (same Delaunay
        face-enumeration step), but each face's appearance value is its own
        minimum-enclosing-ball radius (Welzl's algorithm) rather than a
        circumradius/Gabriel test -- so, unlike Alpha, no coface propagation is
        needed.

        Args:
            points: (N, D) array of point coordinates.
            top_simplices: (M, D + 1) array of Delaunay top simplices (vertex indices).
            max_dim: Maximum simplex dimension to build.
            analyze_manifolds: If True, run the per-threshold manifold analysis in Julia.
            n_samples: If given, select this many thresholds from distinct values.
            eps_max: If given, maximum filtration value to include.
            verify_manifold_only_at_betti_change: If True, check manifold criteria only when Betti numbers change.
            track_connected_components: If True, track connected components.

        Returns:
            A dict with the same keys as ``compute_alpha_filtration``'s payload.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        return self._compute_delaunay_restricted_filtration(
            "compute_delaunay_cech_filtration", points, top_simplices, max_dim,
            analyze_manifolds, n_samples, eps_max,
            verify_manifold_only_at_betti_change, track_connected_components,
        )

    def compute_delaunay_rips_simplices(
        self, points: np.ndarray, top_simplices: np.ndarray, max_dim: int
    ) -> dict:
        """Delaunay-restricted Rips simplices and longest-edge values via Julia backend.

        Lighter counterpart to ``compute_delaunay_rips_filtration``: returns every
        face (up to ``max_dim``) of the Delaunay triangulation given by
        ``top_simplices``, tagged with its Rips (longest-edge) appearance value,
        without running persistence -- for callers building a ``.filtration`` map
        directly (``SimplicialComplex.from_delaunay_rips``) rather than a barcode.

        Args:
            points: (N, D) array of point coordinates.
            top_simplices: (M, D + 1) array of Delaunay top simplices.
            max_dim: Maximum simplex dimension to return.

        Returns:
            A dict mapping each simplex (sorted vertex tuple) to its Rips value.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            cliques, vals = self.backend.compute_delaunay_rips_simplices_jl(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(top_simplices, dtype=np.int64),
                int(max_dim),
            )
            simplices = [tuple(sorted(int(x) for x in s)) for s in cliques]
            values = [float(v) for v in vals]
            return dict(zip(simplices, values))
        except Exception as e:
            raise RuntimeError(f"compute_delaunay_rips_simplices failed: {e!r}")

    def compute_delaunay_cech_simplices(
        self, points: np.ndarray, top_simplices: np.ndarray, max_dim: int
    ) -> dict:
        """Delaunay-restricted Cech simplices and min-enclosing-ball values via Julia backend.

        Lighter counterpart to ``compute_delaunay_cech_filtration``: returns every
        face (up to ``max_dim``) of the Delaunay triangulation given by
        ``top_simplices``, tagged with its Cech (min-enclosing-ball radius)
        appearance value, without running persistence -- for callers building a
        ``.filtration`` map directly (``SimplicialComplex.from_delaunay_cech``)
        rather than a barcode.

        Args:
            points: (N, D) array of point coordinates.
            top_simplices: (M, D + 1) array of Delaunay top simplices.
            max_dim: Maximum simplex dimension to return.

        Returns:
            A dict mapping each simplex (sorted vertex tuple) to its Cech value.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            cliques, vals = self.backend.compute_delaunay_cech_simplices_jl(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(top_simplices, dtype=np.int64),
                int(max_dim),
            )
            simplices = [tuple(sorted(int(x) for x in s)) for s in cliques]
            values = [float(v) for v in vals]
            return dict(zip(simplices, values))
        except Exception as e:
            raise RuntimeError(f"compute_delaunay_cech_simplices failed: {e!r}")

    def compute_rips_cohomology(
        self,
        points: np.ndarray,
        epsilon: float,
        max_dim: int,
        analyze_manifolds: bool = False,
        n_samples: Optional[int] = None,
        verify_manifold_only_at_betti_change: bool = False,
        track_connected_components: bool = False,
    ) -> dict:
        """Implicit persistent COHOMOLOGY of a Vietoris-Rips filtration in Julia.

        Ripser-style engine: simplices are indexed by the combinatorial number
        system and their cofacets enumerated on the fly, so the boundary matrix is
        never assembled. Reduction is the coboundary column reduction with clearing
        and (emergent-pair) owner-recompute. Persistent cohomology yields the
        IDENTICAL barcode to homology (de Silva-Morozov-Vejdemo-Johansson duality),
        so the result is exact -- validated bar-for-bar against
        :meth:`compute_rips_filtration`.

        Returns the same payload dict as :meth:`compute_rips_filtration`

        Args:
            points: (N, D) array of point coordinates.
            epsilon: Maximum edge length (diameter cap) for the complex.
            max_dim: Maximum simplex dimension; H_d computed for d in 0..max_dim.
            analyze_manifolds: If True, run the per-threshold manifold analysis in Julia.
            n_samples: If given, select this many thresholds from distinct values.
            verify_manifold_only_at_betti_change: If True, check manifold criteria only when Betti numbers change.
            track_connected_components: If True, track connected components.

        Returns:
            The payload dict described above.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            res = self.backend.compute_rips_cohomology(
                np.ascontiguousarray(points, dtype=np.float64),
                float(epsilon),
                int(max_dim),
                bool(analyze_manifolds),
                n_samples,
                bool(verify_manifold_only_at_betti_change),
                bool(track_connected_components),
            )
            bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val, _dim_count, total = res[0:8]
            dims = np.asarray(bar_dim, dtype=np.int64).tolist()
            births = np.asarray(bar_birth, dtype=np.float64).tolist()
            deaths = np.asarray(bar_death, dtype=np.float64).tolist()
            barcode = list(zip(dims, births, deaths))

            ids = np.asarray(dim_ids, dtype=np.int64).tolist()
            firsts = np.asarray(dim_first_val, dtype=np.float64).tolist()
            payload = {
                "barcode": barcode,
                "eps_values": np.asarray(eps_values, dtype=np.float64).tolist(),
                "dim_first_appear": {int(d): float(f) for d, f in zip(ids, firsts)},
                "total": int(total),
            }

            if len(res) > 8 and res[8]:
                payload["manifold_data"] = {
                    "epsilons": np.asarray(res[9], dtype=np.float64).tolist(),
                    "is_manifold": np.asarray(res[10], dtype=bool).tolist(),
                    "dimensions": np.asarray(res[11], dtype=np.int64).tolist(),
                    "is_closed": np.asarray(res[12], dtype=bool).tolist(),
                    "failures": np.asarray(res[13], dtype=np.int64).tolist(),
                }
                if len(res) > 14 and len(res[14]) > 0:
                    comp_keys = [np.asarray(x, dtype=np.int64).tolist() for x in res[14]]
                    comp_vals = [list(x) for x in res[15]]
                    payload["component_data"] = [dict(zip(k, v)) for k, v in zip(comp_keys, comp_vals)]
                else:
                    payload["component_data"] = None
            else:
                payload["manifold_data"] = None
                payload["component_data"] = None

            return payload
        except Exception as e:
            raise RuntimeError(f"compute_rips_cohomology failed: {e!r}")

    def simplify_jl(self, simplices: list[tuple]) -> tuple[list[tuple], dict[int, list[int]], dict[tuple, list[tuple]]]:
        """Executes high-performance topology-preserving simplification in Julia.

        Args:
            simplices: A list of tuples representing all simplices in the complex.

        Returns:
            A tuple of (remaining_simplices_list, new_to_original_vertex_map, new_to_original_simplex_map).
        """
        self.require_julia()
        try:
            simplices_jl, v_map_jl, s_map_jl = self.backend.simplify_jl(simplices)
            
            # v_map_jl is Dict{Int, Vector{Int}}
            v_map_py = {int(k): [int(v) for v in val] for k, val in dict(v_map_jl).items()}
            
            # s_map_jl is Dict{Tuple, Vector{Tuple}}
            s_map_py = {tuple(sorted(int(v) for v in k)): [tuple(sorted(int(v) for v in o)) for o in o_list]
                        for k, o_list in dict(s_map_jl).items()}
            
            return list(simplices_jl), v_map_py, s_map_py
        except Exception as e:
            raise RuntimeError(f"simplify_jl failed: {e!r}")
    def compute_alpha_threshold_emst(self, points: np.ndarray, simplices: np.ndarray) -> float:
        """Compute the EMST-based alpha threshold for a point cloud.

        Args:
            points: (N, D) array of point coordinates.
            simplices: (M, D+1) array of Delaunay simplices.

        Returns:
            The squared alpha threshold.
        """
        self.require_julia()
        try:
            return float(self.backend.compute_alpha_threshold_emst_jl(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64)
            ))
        except Exception as e:
            raise RuntimeError(f"compute_alpha_threshold_emst failed: {e!r}")

    def compute_crust_simplices(self, points: np.ndarray, combined_simplices: np.ndarray, n_pts_orig: int) -> list[tuple[int, ...]]:
        """Extract Crust simplices via Julia backend.

        Args:
            points: Combined array of points and Voronoi vertices.
            combined_simplices: Delaunay simplices of combined set.
            n_pts_orig: Number of original points.

        Returns:
            A list of simplex tuples.
        """
        self.require_julia()
        try:
            res = self.backend.compute_crust_simplices_jl(
                np.asarray(points, dtype=np.float64),
                np.asarray(combined_simplices, dtype=np.int64),
                int(n_pts_orig)
            )
            return [tuple(sorted(int(v) for v in s)) for s in res]
        except Exception as e:
            raise RuntimeError(f"compute_crust_simplices failed: {e!r}")

    def compute_voronoi_poles(
        self, points: np.ndarray, vor_vertices: np.ndarray, cell_vertex_lists: list
    ) -> tuple:
        """Per-point Voronoi pole selection (Amenta-Bern) via Julia backend.

        Lighter counterpart wiring for ``estimate_voronoi_poles``: the Voronoi
        diagram itself is computed by ``scipy.spatial.Voronoi`` in Python (no
        Julia Voronoi implementation exists); only the per-point argmax-distance
        positive-pole and opposite-side negative-pole numeric loop runs in Julia.

        Args:
            points: (N, 3) array of the original (unpadded) point coordinates.
            vor_vertices: (V, 3) array, the full Voronoi diagram's vertex
                coordinates (``scipy.spatial.Voronoi.vertices``).
            cell_vertex_lists: Length-N list of lists; ``cell_vertex_lists[i]``
                is the (already ``-1``-filtered) Voronoi vertex indices bounding
                point ``i``'s cell.

        Returns:
            A tuple ``(positive_pole, negative_pole, has_negative_pole, normal,
            pole_radius, status)`` of NumPy arrays, shapes ``(N, 3)``, ``(N, 3)``,
            ``(N,)``, ``(N, 3)``, ``(N,)``, ``(N,)`` respectively. ``status`` is
            0 (pole and opposite pole both found), 1 (empty cell), 2
            (degenerate/zero-radius cell), or 3 (pole found, no opposite pole).

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            flat, offsets = self._flatten_simplices(cell_vertex_lists)
            pos, neg, has_neg, normal, radius, status = self.backend.compute_voronoi_poles_jl(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(vor_vertices, dtype=np.float64),
                flat,
                offsets,
            )
            return (
                np.asarray(pos, dtype=np.float64),
                np.asarray(neg, dtype=np.float64),
                np.asarray(has_neg, dtype=bool),
                np.asarray(normal, dtype=np.float64),
                np.asarray(radius, dtype=np.float64),
                np.asarray(status, dtype=np.int8),
            )
        except Exception as e:
            raise RuntimeError(f"compute_voronoi_poles failed: {e!r}")

    def compute_cocone_filter(
        self,
        points: np.ndarray,
        tetrahedra: np.ndarray,
        normals: np.ndarray,
        pole_radius: np.ndarray,
        theta: float,
        reach_fraction: float,
    ) -> tuple:
        """Cocone angle + pole-reach filter (Amenta-Choi-Dey-Leekha) via Julia backend.

        Fused counterpart to ``cocone_filter``: builds the facet-to-tetrahedra
        map, computes every tetrahedron's circumcenter, and runs the angle and
        reach filters, all in one Julia call.

        Args:
            points: (N + S, 3) array of point coordinates (original points
                followed by any sentinels), as in ``cocone_filter``.
            tetrahedra: (M, 4) int array of Delaunay tetrahedra over ``points``.
            normals: (N, 3) array of per-point pole normal estimates.
            pole_radius: (N,) array of per-point pole radii.
            theta: Cocone half-angle.
            reach_fraction: Minimum pole-radius fraction the dual edge must reach.

        Returns:
            A tuple ``(surviving_triangles, n_candidates)``: a list of 0-based
            ``(v0, v1, v2)`` vertex-index tuples, and the total number of
            distinct all-original-vertex candidate facets considered.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            surviving, n_candidates = self.backend.compute_cocone_filter_jl(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(tetrahedra, dtype=np.int64),
                np.ascontiguousarray(normals, dtype=np.float64),
                np.ascontiguousarray(pole_radius, dtype=np.float64),
                float(theta),
                float(reach_fraction),
            )
            triangles = [tuple(sorted(int(v) for v in s)) for s in surviving]
            return triangles, int(n_candidates)
        except Exception as e:
            raise RuntimeError(f"compute_cocone_filter failed: {e!r}")

    def compute_prune_and_walk(
        self, points: np.ndarray, triangles: np.ndarray, normals: np.ndarray
    ) -> tuple:
        """Per-vertex prune-and-walk manifold-forcing step via Julia backend.

        Fused, per-vertex-parallel counterpart to ``prune_and_walk``: each
        vertex's local link graph, the fast-path single-path-or-cycle check,
        and the tangent-plane angular walk, all in one (per-vertex threaded)
        Julia call.

        Args:
            points: (N, 3) array of point coordinates.
            triangles: (T, 3) int array of candidate triangles (0-based
                vertex indices), typically ``cocone_filter``'s surviving
                triangles.
            normals: (N, 3) array of per-point pole normal estimates.

        Returns:
            A tuple ``(vertex_ids, status, n_kept_edges, kept_flat,
            kept_offsets)`` -- see ``compute_prune_and_walk_jl``'s docstring
            for the exact per-vertex status code meanings and the CSR
            (``kept_flat``/``kept_offsets``) convention for each vertex's
            locally-kept triangles.

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            vertex_ids, status, n_kept_edges, kept_flat, kept_offsets = self.backend.compute_prune_and_walk_jl(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(triangles, dtype=np.int64),
                np.ascontiguousarray(normals, dtype=np.float64),
            )
            return (
                np.asarray(vertex_ids, dtype=np.int64),
                np.asarray(status, dtype=np.int8),
                np.asarray(n_kept_edges, dtype=np.int64),
                np.asarray(kept_flat, dtype=np.int64),
                np.asarray(kept_offsets, dtype=np.int64),
            )
        except Exception as e:
            raise RuntimeError(f"compute_prune_and_walk failed: {e!r}")

    def compute_witness_complex_simplices(self, points: np.ndarray, landmarks_idx: np.ndarray, alpha: float, max_dim: int) -> list[tuple[int, ...]]:
        """Compute Witness Complex 1-skeleton via Julia backend.

        Args:
            points: (N, D) array of point coordinates.
            landmarks_idx: Indices of landmark points.
            alpha: Relaxation parameter.
            max_dim: Maximum dimension.

        Returns:
            A list of simplex tuples.
        """
        self.require_julia()
        try:
            res = self.backend.compute_witness_complex_simplices_jl(
                np.asarray(points, dtype=np.float64),
                np.asarray(landmarks_idx, dtype=np.int64),
                float(alpha),
                int(max_dim)
            )
            return [tuple(sorted(int(v) for v in s)) for s in res]
        except Exception as e:
            raise RuntimeError(f"compute_witness_complex_simplices failed: {e!r}")

    def compute_vietoris_rips(self, points: np.ndarray, epsilon: float, max_dim: int) -> list[tuple[int, ...]]:
        """Compute Vietoris-Rips complex via Julia backend.

        Args:
            points: (N, D) array of point coordinates.
            epsilon: Distance threshold.
            max_dim: Maximum dimension.

        Returns:
            A list of simplex tuples.
        """
        self.require_julia()
        try:
            res = self.backend.compute_vietoris_rips(
                np.asarray(points, dtype=np.float64),
                float(epsilon),
                int(max_dim)
            )
            return [tuple(sorted(int(v) for v in s)) for s in res]
        except Exception as e:
            raise RuntimeError(f"compute_vietoris_rips failed: {e!r}")

    def compute_vietoris_rips_from_distance_matrix(
        self, distance_matrix: np.ndarray, epsilon: float, max_dim: int
    ) -> list[tuple[int, ...]]:
        """Compute a Vietoris-Rips complex from a precomputed distance matrix via Julia backend.

        Unlike ``compute_vietoris_rips`` (which computes pairwise distances from point
        coordinates), this reads distances directly from an (N, N) matrix -- for a
        precomputed or non-Euclidean distance matrix, or simply to avoid re-deriving
        distances a caller already has, without paying the O(N^2) NumPy edge-extraction
        cost of the pure-Python path on large N.

        Args:
            distance_matrix: (N, N) symmetric distance matrix.
            epsilon: Distance threshold.
            max_dim: Maximum dimension.

        Returns:
            A list of simplex tuples.
        """
        self.require_julia()
        try:
            res = self.backend.compute_vietoris_rips_from_distance_matrix(
                np.ascontiguousarray(distance_matrix, dtype=np.float64),
                float(epsilon),
                int(max_dim)
            )
            return [tuple(sorted(int(v) for v in s)) for s in res]
        except Exception as e:
            raise RuntimeError(f"compute_vietoris_rips_from_distance_matrix failed: {e!r}")

    def is_homology_manifold_jl(self, simplices: list[list[int]], max_dim: int) -> tuple[bool, int, dict[int, str]]:
        """Accelerated manifold certification in Julia.

        Args:
            simplices: A list of simplex vertex index lists.
            max_dim: The maximum dimension of simplices to consider.

        Returns:
            A tuple of (is_manifold, dimension, diagnostics_dict).

        Raises:
            RuntimeError: If the Julia call fails.
        """
        self.require_julia()
        try:
            is_manifold, dimension, diagnostics_jl = self.backend.is_homology_manifold_jl(
                simplices,
                int(max_dim)
            )
            # diagnostics_jl is a Dict{Int, String} from Julia
            diagnostics = {int(k): str(v) for k, v in dict(diagnostics_jl).items()}
            return bool(is_manifold), int(dimension), diagnostics
        except Exception as e:
            raise RuntimeError(f"is_homology_manifold_jl failed: {e!r}")

    def compute_discrete_morse_gradient_jl(self, simplices: list[list[int]]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Accelerated Discrete Morse matching in Julia.

        Args:
            simplices: A list of simplex vertex index lists.

        Returns:
            A list of pairs ((sigma), (tau)) representing the gradient matching.
        """
        self.require_julia()
        try:
            res = self.backend.compute_discrete_morse_gradient_jl(simplices)
            # res is Vector{Vector{Vector{Int64}}} from Julia
            return [(tuple(sorted(int(x) for x in pair[0])), tuple(sorted(int(x) for x in pair[1]))) for pair in res]
        except Exception as e:
            raise RuntimeError(f"compute_discrete_morse_gradient_jl failed: {e!r}")

    # ──────────────────────────────────────────────────────────────────────
    # Proposal 5 (REVISED): Bounded Controlled Cohomology bridge methods
    # ──────────────────────────────────────────────────────────────────────

    def todd_coxeter_index(
        self, generators: list[str], relations: list[str], max_index: int
    ) -> tuple[bool, int, np.ndarray]:
        """Run Todd-Coxeter coset enumeration over the trivial subgroup.

        Args:
            generators: Generator name strings.
            relations: Space-separated relator strings (each token is a
                generator or `gen^N` for integer N).
            max_index: Maximum number of cosets to enumerate.

        Returns:
            Tuple `(converged, n_cosets, coset_table)` where `coset_table` is
            an int64 array of shape `(n_cosets, 2*n_gens)`.
        """
        self.require_julia()
        converged, n_cosets, table = self.backend.todd_coxeter_index_jl(
            list(generators), list(relations), int(max_index)
        )
        n_cosets = int(n_cosets)
        if not bool(converged):
            return False, n_cosets, np.zeros((0, 0), dtype=np.int64)
        table_np = np.array(table, dtype=np.int64)
        return True, n_cosets, table_np

    def cayley_table(
        self, coset_table: np.ndarray, generators: list[str]
    ) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
        """Build the Cayley table of a finite group from a Todd-Coxeter table.

        Args:
            coset_table: Output from `todd_coxeter_index`; shape
                `(|G|, 2*n_gens)`.
            generators: Generator name strings.

        Returns:
            Tuple `(cayley, inverse_indices, id_idx, words)` with all indices
            **1-based** to match the Julia kernel's convention. `cayley[i, j]`
            (1-based) is the index of the product of group elements i and j;
            `inverse_indices[i]` (1-based) is the index of the inverse.
        """
        self.require_julia()
        cayley_jl, inverse_jl, id_idx, words_jl = self.backend.cayley_table_jl(
            np.asarray(coset_table, dtype=np.int64), list(generators)
        )
        cayley = np.array(cayley_jl, dtype=np.int64)
        inverse = np.array(inverse_jl, dtype=np.int64)
        words = [str(w) for w in words_jl]
        return cayley, inverse, int(id_idx), words

    def sphere_recognition_pl(self, simplices: list[list[int]], dim: int) -> tuple[bool, str]:
        """Perform PL sphere recognition in Julia.

        Returns:
            (is_sphere, reason)
        """
        self.require_julia()
        res = self.backend.sphere_recognition_pl(simplices, int(dim))
        # res is a tuple (Bool, String)
        return bool(res[0]), str(res[1])

    def surgery_relative_boundary_sparse(
        self, K_q: list[list[int]], K_qp1: list[list[int]], Kb_indices: list[int]
    ):
        """Build relative boundary matrix in Julia."""
        self.require_julia()
        return self.backend.surgery_relative_boundary_sparse(
            K_q, K_qp1, [int(idx) + 1 for idx in Kb_indices]  # Julia uses 1-based
        )

    def linking_seifert_solve_z(self, B, b: np.ndarray) -> tuple[np.ndarray, bool, str]:
        """Solve B*f = b over Z in Julia using SNF."""
        self.require_julia()
        f_jl, success, reason = self.backend.linking_seifert_solve_z(
            B, np.asarray(b, dtype=np.int64)
        )
        return np.array(f_jl, dtype=np.int64), bool(success), str(reason)

    def linking_gauss_riemann(
        self,
        Ka_starts: np.ndarray,
        Ka_ends: np.ndarray,
        Ka_mult: np.ndarray,
        Kb_starts: np.ndarray,
        Kb_ends: np.ndarray,
        Kb_mult: np.ndarray,
        n_samples: int = 24,
    ) -> float:
        """Compute the Gauss linking integral lk(K_a, K_b) in Julia.

        `Ka_starts`/`Ka_ends` are (N_a, 3) arrays of oriented edge endpoints;
        `Ka_mult` is (N_a,) integer multiplicities. Same shape for K_b.
        Returns the value already divided by 4π (caller rounds to integer).
        """
        self.require_julia()
        Ka_s = np.ascontiguousarray(Ka_starts, dtype=np.float64)
        Ka_e = np.ascontiguousarray(Ka_ends, dtype=np.float64)
        Ka_m = np.ascontiguousarray(Ka_mult, dtype=np.int64)
        Kb_s = np.ascontiguousarray(Kb_starts, dtype=np.float64)
        Kb_e = np.ascontiguousarray(Kb_ends, dtype=np.float64)
        Kb_m = np.ascontiguousarray(Kb_mult, dtype=np.int64)
        if Ka_s.size == 0 or Kb_s.size == 0:
            return 0.0
        val = self.backend.linking_gauss_riemann_jl(
            Ka_s, Ka_e, Ka_m, Kb_s, Kb_e, Kb_m, int(n_samples)
        )
        return float(val)

    def linking_intersection_pairing(
        self,
        a: np.ndarray,
        f: np.ndarray,
        Cp: list[list[int]],
        Cqp1: list[list[int]],
        n: int,
    ) -> int:
        """Compute simplicial intersection pairing in Julia."""
        self.require_julia()
        val = self.backend.linking_intersection_pairing(
            np.asarray(a, dtype=np.int64),
            np.asarray(f, dtype=np.int64),
            Cp,
            Cqp1,
            int(n),
        )
        return int(val)

    def linking_intersection_batch(
        self,
        a_series: list[np.ndarray],
        f: np.ndarray,
        Cp: list[list[int]],
        Cqp1: list[list[int]],
        n: int,
    ) -> np.ndarray:
        """Compute ⟨K_a_i, F⟩ for each a-vector in a_series, reusing precomputed Seifert chain f.

        Args:
            a_series: List of a-vectors (one per unlink pass). Each is a 1-D int64 array.
            f:        Precomputed Seifert chain (solution to B·f = b_Kb).
            Cp:       p-simplices of K (ambient), as list-of-lists.
            Cqp1:     (q+1)-simplices of K (ambient), as list-of-lists.
            n:        Ambient complex dimension.

        Returns:
            np.ndarray of shape (len(a_series),) with dtype int64 — one lk per a-vector.
        """
        self.require_julia()
        a_series_converted = [np.asarray(a, dtype=np.int64) for a in a_series]
        result = self.backend.linking_intersection_batch(
            a_series_converted,
            np.asarray(f, dtype=np.int64),
            Cp,
            Cqp1,
            int(n),
        )
        return np.array(result, dtype=np.int64)

    def linking_intersect_2chains(
        self,
        F_a: np.ndarray,
        F_b: np.ndarray,
        simplices_1: list[list[int]],
        simplices_2: list[list[int]],
        simplices_3: list[list[int]],
    ) -> np.ndarray:
        """Compute the intersection number of two 2-chains via the Julia backend.

        Args:
            F_a: First 2-chain coefficient vector as int64 ndarray.
            F_b: Second 2-chain coefficient vector as int64 ndarray.
            simplices_1: 1-simplices of the underlying complex.
            simplices_2: 2-simplices of the underlying complex.
            simplices_3: 3-simplices of the underlying complex.

        Returns:
            np.ndarray with dtype int64 holding the intersection result.
        """
        self.require_julia()
        result = self.backend.linking_intersect_2chains(
            np.asarray(F_a, dtype=np.int64),
            np.asarray(F_b, dtype=np.int64),
            simplices_1,
            simplices_2,
            simplices_3,
        )
        return np.array(result, dtype=np.int64)

    def alexander_from_seifert(self, V: np.ndarray) -> dict | None:
        """Compute Alexander polynomial det(tV - V^T) via Julia's AbstractAlgebra.

        Args:
            V: Seifert matrix as 2D int64 ndarray.

        Returns:
            dict {degree: coeff} or None on failure.
        """
        self.require_julia()
        V_arr = np.asarray(V, dtype=np.int64)
        rows = [list(V_arr[i]) for i in range(V_arr.shape[0])]
        coeffs_jl, min_deg = self.backend.alexander_from_seifert_jl(rows)
        coeffs = list(coeffs_jl)
        result = {}
        for k, c in enumerate(coeffs):
            d = int(min_deg) + k
            if int(c) != 0:
                result[d] = int(c)
        return result if result else {0: 1}

    def knot_signature(self, V: np.ndarray) -> int:
        """Compute knot signature σ(K) = sig(V + V^T) via Julia.

        Args:
            V: Seifert matrix as 2D int64 ndarray.

        Returns:
            int (#positive - #negative eigenvalues of V + V^T).
        """
        self.require_julia()
        V_arr = np.asarray(V, dtype=np.int64)
        rows = [list(V_arr[i]) for i in range(V_arr.shape[0])]
        return int(self.backend.knot_signature_jl(rows))

    def compute_cohomology_basis_jl(
        self,
        D_m,
        D_mp1,
        D_n,
    ) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
        """Compute free H^m cohomology generators from sparse boundary matrices via Julia.

        Replaces the Python path of smith_normal_decomp + SymPy.inv in IntersectionForm.from_complex
        for the case β_m > 50, where SymPy matrix inversions become the bottleneck.

        Args:
            D_m:   scipy sparse boundary matrix ∂_m (nrows_Dm × ncols_Dm).
            D_mp1: scipy sparse boundary matrix ∂_{m+1} (nrows_Dmp1 × ncols_Dmp1).
            D_n:   scipy sparse boundary matrix ∂_n (nrows_Dn × ncols_Dn).

        Returns:
            (cocycles, n_rows, n_cols, free_col_indices, F) where:
              - cocycles: (n_rows × n_cols) ndarray — full cocycle matrix.
              - free_col_indices: 0-based int64 array of free-generator columns.
              - F: fundamental class as int64 vector of length ncols_Dn.
        """
        self.require_julia()
        import scipy.sparse as sp_mod

        def _to_coo(mat, expected_shape):
            if mat is None or mat.shape[0] == 0 or mat.shape[1] == 0:
                nr, nc = expected_shape
                return (
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                    nr,
                    nc,
                )
            coo = sp_mod.coo_matrix(mat.astype(np.int64))
            return (
                coo.row.astype(np.int64),
                coo.col.astype(np.int64),
                coo.data.astype(np.int64),
                int(coo.shape[0]),
                int(coo.shape[1]),
            )

        Dm_I, Dm_J, Dm_V, nrDm, ncDm     = _to_coo(D_m,   D_m.shape   if D_m is not None else (0, 0))
        Dmp1_I, Dmp1_J, Dmp1_V, nrDmp1, ncDmp1 = _to_coo(D_mp1, D_mp1.shape if D_mp1 is not None else (0, 0))
        Dn_I, Dn_J, Dn_V, nrDn, ncDn     = _to_coo(D_n,   D_n.shape   if D_n is not None else (0, 0))

        cocycles_flat, n_rows, n_cols, free_cols, F = self.backend.compute_cohomology_basis_jl(
            Dm_I, Dm_J, Dm_V, nrDm, ncDm,
            Dmp1_I, Dmp1_J, Dmp1_V, nrDmp1, ncDmp1,
            Dn_I, Dn_J, Dn_V, nrDn, ncDn,
        )
        n_rows = int(n_rows)
        n_cols = int(n_cols)
        cocycles_np = np.array(cocycles_flat, dtype=np.int64).reshape(n_rows, n_cols, order="F")
        free_cols_np = np.array(free_cols, dtype=np.int64)
        F_np = np.array(F, dtype=np.int64)
        return cocycles_np, n_rows, n_cols, free_cols_np, F_np

    def surgery_handle_attach(
        self,
        K_simplices: dict[int, list[list[int]]],
        attaching_sphere: list[list[int]],
        tubular_neighborhood: list[list[int]],
        co_disk_simplices: list[list[int]],
        vertex_offset: int,
        index_k: int,
        ambient_dim: int,
    ) -> dict:
        """Perform handle attachment in Julia."""
        self.require_julia()
        res = self.backend.surgery_handle_attach(
            K_simplices,
            attaching_sphere,
            tubular_neighborhood,
            co_disk_simplices,
            int(vertex_offset),
            int(index_k),
            int(ambient_dim),
        )
        # Convert the Julia Dict{Int, Vector{Vector{Int}}} back to a Python dict.
        # (Do not use `pyconvert(dict, res)`: pyconvert's first argument must be a
        # Julia *type*, not the Python `dict` object.)
        return {int(d): [[int(v) for v in s] for s in simps] for d, simps in res.items()}

    def cayley_convolve(
        self, a: np.ndarray, b: np.ndarray, cayley: np.ndarray
    ) -> np.ndarray:
        """Group-ring multiplication via Cayley-table convolution."""
        self.require_julia()
        res = self.backend.cayley_convolve_jl(
            np.asarray(a, dtype=np.int64),
            np.asarray(b, dtype=np.int64),
            np.asarray(cayley, dtype=np.int64),
        )
        return np.array(res, dtype=np.int64)

    def lift_boundary_to_cover(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        group_indices: np.ndarray,
        coeffs: np.ndarray,
        n_g: int,
        m_base: int,
        n_base: int,
        cayley: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lift Z[G]-boundary data to the universal cover as COO triples.

        All input indices are 1-based; outputs are 1-based as well — Python
        callers should subtract 1 to obtain 0-based scipy.sparse indices.
        """
        self.require_julia()
        out_rows, out_cols, out_vals = self.backend.lift_boundary_to_cover_jl(
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(group_indices, dtype=np.int64),
            np.asarray(coeffs, dtype=np.int64),
            int(n_g),
            int(m_base),
            int(n_base),
            np.asarray(cayley, dtype=np.int64),
        )
        return (
            np.array(out_rows, dtype=np.int64),
            np.array(out_cols, dtype=np.int64),
            np.array(out_vals, dtype=np.int64),
        )

    def fox_derivative_block(
        self,
        relator: list[int],
        gen_idx: int,
        cayley: np.ndarray,
        gen_to_group: list[int],
        inverse_indices: np.ndarray,
        rho_images: list[np.ndarray],
        degree: int,
        complex_dtype: bool = False,
    ) -> np.ndarray:
        """Compute a Fox-derivative block ∂w/∂g_i evaluated through ρ.

        Args:
            relator: Relator word as a list of signed generator indices
                (1-based; positive = generator, negative = inverse).
            gen_idx: 1-based generator index to differentiate by.
            cayley: Cayley table (1-based indices), int64.
            gen_to_group: For each generator (1..n_gens), its 1-based index
                in the Cayley table.
            inverse_indices: For each group element, its 1-based inverse index.
            rho_images: List `[ρ(g_1), …, ρ(g_n)]` where each `ρ(g_k)` is
                a `(degree, degree)` numpy array.
            degree: Representation degree.
            complex_dtype: If True, evaluate in `complex128`; else `float64`.
        """
        self.require_julia()
        if complex_dtype:
            rho_packed = [np.asarray(R, dtype=np.complex128) for R in rho_images]
            block = self.backend.fox_derivative_block_complex_jl(
                list(int(s) for s in relator),
                int(gen_idx),
                np.asarray(cayley, dtype=np.int64),
                list(int(g) for g in gen_to_group),
                np.asarray(inverse_indices, dtype=np.int64),
                rho_packed,
                int(degree),
            )
            return np.array(block, dtype=np.complex128)
        rho_packed = [np.asarray(R, dtype=np.float64) for R in rho_images]
        block = self.backend.fox_derivative_block_real_jl(
            list(int(s) for s in relator),
            int(gen_idx),
            np.asarray(cayley, dtype=np.int64),
            list(int(g) for g in gen_to_group),
            np.asarray(inverse_indices, dtype=np.int64),
            rho_packed,
            int(degree),
        )
        return np.array(block, dtype=np.float64)


    def compute_hodge_harmonics(self, L: sp.csr_matrix, b_k: int) -> np.ndarray:
        """Compute the exact harmonic basis of the sparse Hodge Laplacian L."""
        self.require_julia()
        coo = sp.coo_matrix(L.astype(np.float64))
        res = self.backend.compute_hodge_harmonics_jl(
            coo.row.astype(np.int64),
            coo.col.astype(np.int64),
            coo.data,
            int(coo.shape[0]),
            int(coo.shape[1]),
            int(b_k)
        )
        return np.array(res, dtype=np.float64)

    def compute_hodge_decomposition(self, B_k: sp.csr_matrix, B_kp1: sp.csr_matrix, L: sp.csr_matrix, chain: np.ndarray, b_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose chain into exact, coexact, and harmonic parts."""
        self.require_julia()
        
        def _to_coo(mat: sp.csr_matrix):
            coo = sp.coo_matrix(mat.astype(np.float64))
            return (
                coo.row.astype(np.int64),
                coo.col.astype(np.int64),
                coo.data,
                int(coo.shape[0]),
                int(coo.shape[1])
            )
            
        Bk_I, Bk_J, Bk_V, Bk_nr, Bk_nc = _to_coo(B_k)
        Bkp1_I, Bkp1_J, Bkp1_V, Bkp1_nr, Bkp1_nc = _to_coo(B_kp1)
        L_I, L_J, L_V, L_nr, L_nc = _to_coo(L)
        
        alpha, beta, h = self.backend.compute_hodge_decomposition_jl(
            Bk_I, Bk_J, Bk_V, Bk_nr, Bk_nc,
            Bkp1_I, Bkp1_J, Bkp1_V, Bkp1_nr, Bkp1_nc,
            L_I, L_J, L_V, L_nr, L_nc,
            np.asarray(chain, dtype=np.float64), int(b_k)
        )
        return (
            np.array(alpha, dtype=np.float64),
            np.array(beta, dtype=np.float64),
            np.array(h, dtype=np.float64),
        )

    # Singleton instance
julia_engine = JuliaBridge()

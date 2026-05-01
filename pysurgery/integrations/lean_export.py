import numpy as np
import re
import shutil
import subprocess
import tempfile
from pydantic import BaseModel, ConfigDict


class LeanCheckResult(BaseModel):
    """Execution diagnostics and status for a formal Lean 4 code check.

    Overview:
        A LeanCheckResult encapsulates the outcome of running a generated Lean script 
        through the `lean` compiler. It tracks availability, process exit codes, 
        and full compiler logs to assist in identifying formal verification failures.

    Key Concepts:
        - **Formal Verification**: Using a proof assistant to ensure mathematical correctness.
        - **Decidability**: Lean's ability to computationally verify matrix identities.

    Common Workflows:
        1. **Generate** -> `generate_lean_isomorphism_certificate()`
        2. **Run** -> `run_lean_check()`
        3. **Analyze** -> Inspect `LeanCheckResult.success` and `stderr`.

    Attributes:
        available (bool): True if the `lean` executable was found on the path.
        exit_code (int): Process exit code from the Lean compiler.
        stdout (str): Output from Lean (typically contains timing/proof info).
        stderr (str): Error output (typically contains proof failure details).
        success (bool): True if the Lean theorem was successfully proved.
        command (str): The exact shell command executed.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    available: bool
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    command: str


def run_lean_check(
    lean_code: str, lean_cmd: str = "lean", timeout_sec: int = 20
) -> LeanCheckResult:
    """Optionally run local Lean on generated code and return structured diagnostics.

    Args:
        lean_code (str): The Lean code to check.
        lean_cmd (str): The command to run Lean. Defaults to "lean".
        timeout_sec (int): Timeout in seconds. Defaults to 20.

    Returns:
        LeanCheckResult: The result of the Lean check.
    """
    exe = shutil.which(lean_cmd)
    if exe is None:
        return LeanCheckResult(
            available=False,
            exit_code=127,
            stdout="",
            stderr=f"Lean executable '{lean_cmd}' not found.",
            success=False,
            command=lean_cmd,
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(lean_code)
        lean_path = f.name

    proc = subprocess.run(
        [exe, lean_path],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return LeanCheckResult(
        available=True,
        exit_code=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
        success=(proc.returncode == 0),
        command=f"{exe} {lean_path}",
    )


def generate_lean_isomorphism_certificate(
    Q1: np.ndarray, Q2: np.ndarray, P: np.ndarray, theorem_name: str = "homeo_cert"
) -> str:
    """Generates a Lean 4 proof script verifying an Algebraic Isomorphism Certificate.

    What is Being Computed?:
        Generates formal Lean 4 source code that proves Pᵀ * Q1 * P = Q2 and 
        det(P) = ±1. This provides a machine-verifiable certificate that two 
        intersection forms (and thus their underlying manifolds) are algebraically 
        isomorphic.

    Algorithm:
        1. Validates that Q1, Q2, and P are square integer matrices of the same dimension.
        2. Converts NumPy arrays into Lean matrix literals (`!![1, 2; 3, 4]`).
        3. Constructs a Lean theorem statement asserting the matrix identity.
        4. Selects the appropriate proof tactic (`decide` for small matrices, 
           `native_decide` for larger ones).

    Preserved Invariants:
        - Algebraic isomorphism of intersection forms is a necessary condition 
          for homeomorphism of simply-connected 4-manifolds.
        - Unimodularity of P ensures the isomorphism is over Z (integral).

    Args:
        Q1: The first intersection form matrix (Z-valued).
        Q2: The second intersection form matrix (Z-valued).
        P: The candidate isomorphism certificate matrix.
        theorem_name: The name for the generated Lean theorem.

    Returns:
        str: Lean 4 source code containing the proof script.

    Use When:
        - Providing high-assurance evidence for a homeomorphism claim.
        - Bridging numerical/algebraic results from Python to formal proof assistants.
        - Fulfilling "Certificate" requirements in safety-critical topological analysis.

    Example:
        lean_code = generate_lean_isomorphism_certificate(Q_A, Q_B, P_isometry)
        with open("certificate.lean", "w") as f:
            f.write(lean_code)
    """

    Q1 = np.asarray(Q1)
    Q2 = np.asarray(Q2)
    P = np.asarray(P)

    if Q1.ndim != 2 or Q2.ndim != 2 or P.ndim != 2:
        raise ValueError("Lean export expects 2D matrices.")
    if (
        Q1.shape[0] != Q1.shape[1]
        or Q2.shape[0] != Q2.shape[1]
        or P.shape[0] != P.shape[1]
    ):
        raise ValueError("Lean export expects square matrices.")
    if Q1.shape != Q2.shape or Q1.shape != P.shape:
        raise ValueError("Q1, Q2, and P must have identical square shape.")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", theorem_name):
        raise ValueError("theorem_name must be a valid Lean identifier.")

    def matrix_to_lean(mat):
        rows = []
        for r in mat:
            vals = []
            for x in r:
                xi = int(x)
                if x != xi:
                    raise ValueError("Lean export expects integer matrices over Z.")
                vals.append(str(xi))
            row_str = "![" + ", ".join(vals) + "]"
            rows.append(row_str)
        return "![" + ",\n  ".join(rows) + "]"

    n = Q1.shape[0]

    tactic = "native_decide" if n > 4 else "decide"

    lean_code = f"""import Mathlib

-- Generated by pysurgery
-- Theorem: The forms Q1 and Q2 are isomorphic via P.

def Q1 : Matrix (Fin {n}) (Fin {n}) ℤ :=
  {matrix_to_lean(Q1)}

def Q2 : Matrix (Fin {n}) (Fin {n}) ℤ :=
  {matrix_to_lean(Q2)}

def P : Matrix (Fin {n}) (Fin {n}) ℤ :=
  {matrix_to_lean(P)}

theorem {theorem_name}_valid : Pᵀ * Q1 * P = Q2 := by
  -- The proof is by strict matrix evaluation (decide)
  {tactic}

theorem {theorem_name}_unimodular : (P.det = 1) ∨ (P.det = -1) := by
  -- Evaluate determinant
  {tactic}
"""
    return lean_code

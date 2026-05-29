import numpy as np
from pysurgery.algebra.exact_sequences import Morphism, ExactSequence, ShortExactSequence

def test_morphism_basic():
    # f: Z^2 -> Z^2, (x,y) -> (2x, 0)
    matrix = np.array([[2, 0], [0, 0]])
    f = Morphism(matrix, 2, 2)
    
    # Kernel should be spanned by (0, 1)
    ker = f.kernel_basis()
    assert ker.shape[1] == 1
    assert np.all(f(ker[:, 0]) == 0)
    
    # Image should be spanned by (2, 0)
    im = f.image_basis()
    assert im.shape[1] == 1
    # Check if (2,0) is in image
    assert f.lift(np.array([2, 0])) is not None
    assert f.lift(np.array([1, 0])) is None # Not integral lift

def test_exactness_simple():
    # Z --[1,0]^T--> Z^2 --[0,1]--> Z
    f = Morphism(np.array([[1], [0]]), 1, 2)
    g = Morphism(np.array([[0, 1]]), 2, 1)
    
    seq = ExactSequence([None, "Z", "Z^2", "Z", None], [f, g])
    
    # verify_exactness(index) checks Im(morphisms[index]) == Ker(morphisms[index+1])
    # But wait, my ExactSequence constructor took [f, g] and my verify_exactness logic
    # uses morphisms[index] and morphisms[index+1].
    
    assert seq.verify_exactness(0)

def test_short_exact_sequence_splitting():
    # 0 -> Z -> Z^2 -> Z -> 0 (splits)
    f = Morphism(np.array([[1], [0]]), 1, 2)
    g = Morphism(np.array([[0, 1]]), 2, 1)
    ses = ShortExactSequence("Z", "Z^2", "Z", f, g)
    
    assert ses.is_split()

    # 0 -> Z --x2--> Z -> Z/2Z -> 0
    # Our framework currently deals with free modules Z^n. 
    # For Z -> Z -> Z/2Z, we'd need to represent Z/2Z as a module with relations.
    # For now, let's test a non-splitting map between free modules if we can find one.
    # Actually, if the sequence is 0 -> A -> B -> C -> 0 and C is free, it always splits.
    
def test_lifting_and_operators():
    # Test element-level calculus
    # f: Z^2 -> Z, (x,y) -> x + y
    f = Morphism(np.array([[1, 1]]), 2, 1)
    
    x = np.array([10, 5])
    y = f(x)
    assert y[0] == 15
    
    # Lift y back to source
    x_lifted = f.lift(y)
    assert np.all(f(x_lifted) == y)
    
    # Mathematical representation
    latex = f.to_latex()
    assert "\\begin{matrix}1 & 1\\end{matrix}" in latex

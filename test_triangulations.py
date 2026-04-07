from pysurgery.integrations.gudhi_bridge import extract_complex_data
from pysurgery.core.complexes import ChainComplex
import gudhi

def build_s3():
    st = gudhi.SimplexTree()
    # Boundary of 4-simplex (5 vertices, 0,1,2,3,4)
    for i in range(5):
        tet = [j for j in range(5) if j != i]
        st.insert(tet)
    return st

def build_t3(nx=3, ny=3, nz=3):
    st = gudhi.SimplexTree()
    def v(i, j, k): return (i % nx) * ny * nz + (j % ny) * nz + (k % nz)
    # Triangulate each cube into 6 tetrahedra
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v000 = v(i, j, k)
                v100 = v(i+1, j, k)
                v010 = v(i, j+1, k)
                v110 = v(i+1, j+1, k)
                v001 = v(i, j, k+1)
                v101 = v(i+1, j, k+1)
                v011 = v(i, j+1, k+1)
                v111 = v(i+1, j+1, k+1)
                
                # Standard Kuhn triangulation of cube
                st.insert([v000, v100, v110, v111])
                st.insert([v000, v100, v101, v111])
                st.insert([v000, v010, v110, v111])
                st.insert([v000, v010, v011, v111])
                st.insert([v000, v001, v101, v111])
                st.insert([v000, v001, v011, v111])
    return st

def print_homology(st):
    boundaries, cells, _, _ = extract_complex_data(st)
    cc = ChainComplex(boundaries=boundaries, dimensions=list(cells.keys()))
    print("Cells:", cells)
    for i in range(len(cells)):
        print(f"H{i}:", cc.homology(i)[0])

print("S3:")
print_homology(build_s3())

print("T3:")
print_homology(build_t3(3,3,3))

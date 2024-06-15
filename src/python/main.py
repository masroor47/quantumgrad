import numpy as np
from matrix_ops_wrapper import add_matrices, matmul_simple


def test_elementwise_add():
    rows = 1<<15
    cols = 1<<15
    rows = int(rows * 1.6)
    print(f"{rows = }")
    shape = (rows, cols)
    a = np.ones(shape).astype(np.float32)
    b = np.ones(shape).astype(np.float32) * 2
    print(f"a in py main: {a}")
    print(f"b in py main: {b}")
    print()
    for _ in range(10):
        c = add_matrices(a, b)
    print(f"c in py main: \n{c}")

def test_matmul_simple():
    lrows = 1<<14
    lcols = 1<<15
    rrows = 1<<15
    rcols = 1<<15

    lshape = (lrows, lcols)
    rshape = (rrows, rcols)
    left = np.ones(lshape).astype(np.float32)
    right = np.ones(rshape).astype(np.float32)

    c = matmul_simple(left, right)

    print(c)
    print(c.shape)


def main():
    test_matmul_simple()

if __name__ == '__main__':
    main()
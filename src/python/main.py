import numpy as np
from matrix_ops_wrapper import add_matrices


def main():
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

if __name__ == '__main__':
    main()
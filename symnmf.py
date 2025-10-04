import sys, random, numpy as np
import symnmfmodule as capi

np.random.seed(1234)

def avg_of_matrix(M):
    """average of all entries in a 2D list """
    total = 0.0
    count = 0
    for row in M:
        for x in row:
            total += x
            count += 1
    return (total / count) if count else 0.0


def init_H(W, k, set_seed=True):
    """initialize H (n x k) """
    if set_seed:
        np.random.seed(1234)
    n = len(W)
    return np.random.uniform(0.0, 1.0, size=(n, k)).tolist()


def solve_symnmf(N, H0):
    """delegate to the c extension """
    return capi.symnmf(N, H0, max_iters=300, eps=1e-4, alpha=1.0)



def symnmf_from_file(filename, k, seed=None):
    """builds N from CSV, init H0  and run symnmf """
    if seed is not None:
        random.seed(int(seed))
    N = capi.norm(filename)
    H0 = init_H(N, k, set_seed=True)
    return solve_symnmf(N, H0)


def print_matrix(mat):
    """prints matrix as CSV with 4 decimals """
    for row in mat:
        line = ",".join(
            ("{:.4f}".format(0.0) if abs(x) < 0.00005 else "{:.4f}".format(x)).replace("-0.0000", "0.0000") # clamp tiny values to 0.0000 and fix -0.0000
            for x in row
        )
        print(line)


def run(goal, filename, k):
    """performs the program demands """
    if goal == "sym":
        M = capi.sym(filename)
        print_matrix(M)
        return 0
    if goal == "ddg":
        M = capi.ddg(filename)
        print_matrix(M)
        return 0
    if goal == "norm":
        M = capi.norm(filename)
        print_matrix(M)
        return 0
    if goal == "symnmf":
        if k <= 1:
            return 1
        # use normalized similarity for factorization
        N = capi.norm(filename)
        H0 = init_H(N, k, set_seed=True)
        H = solve_symnmf(N, H0)
        print_matrix(H)
        return 0
    return 1


def main():
    try:
        if len(sys.argv) != 4:
            print("An Error Has Occurred")
            return
        k_str, goal, filename = sys.argv[1], sys.argv[2], sys.argv[3]
        try:
            k = int(k_str)
        except Exception:
            print("An Error Has Occurred")
            return
        rc = run(goal, filename, k)
        if rc != 0:
            print("An Error Has Occurred")
    except Exception:
        print("An Error Has Occurred")


if __name__ == "__main__":
    main()
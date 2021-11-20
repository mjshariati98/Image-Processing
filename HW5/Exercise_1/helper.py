from scipy.sparse.linalg import spsolve
import numpy as np
from scipy import sparse


def build_A_matrix(n, m):
    """
    Build the A matrix for solving poisson equation. use this wikipedia page:
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    :param n:
    :param m:
    :return:
    """
    # D = build_D_matrix(m)
    # Eye_matrix = np.eye(m, m)
    #
    # A = np.zeros((m * n, m * n))
    # # put D in A diagonal
    # for i in range(n):
    #     A[i:i + m, i:i + m] = D
    #
    # # put -I's in A
    # for i in range(n - 1):
    #     A[i * m:i * m + m, (i + 1) * m:(i + 1) * m + m] = -Eye_matrix
    #     A[(i + 1) * m:(i + 1) * m + m, i * m:i * m + m] = -Eye_matrix
    # this way doesn't work! memory error!
    # after search at internet, I understand I must use scipy.sparse for building matrices that use compressing methods.

    D = build_D_matrix(m)
    Eye_matrix = np.eye(m, m)
    A = sparse.lil_matrix((m * n, m * n))
    # put D in A diagonal
    for i in range(n):
        A[i * m:i * m + m, i * m:i * m + m] = D
    # put -I's in A
    for i in range(n - 1):
        A[i * m:i * m + m, (i + 1) * m:(i + 1) * m + m] = -Eye_matrix
        A[(i + 1) * m:(i + 1) * m + m, i * m:i * m + m] = -Eye_matrix

    return A


def build_D_matrix(m):
    # D = np.zeros((m, m))
    # for i in range(m):
    #     for j in range(m):
    #         if i == j:
    #             D[i, j] = 4
    #         elif i == j + 1 or i == j - 1:
    #             D[i, j] = -1
    # this way doesn't work! memory error!
    # after search at internet, I understand I must use scipy.sparse for building matrices that use compressing methods.

    D = sparse.lil_matrix((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                D[i, j] = 4
            elif i == j + 1 or i == j - 1:
                D[i, j] = -1
    return D


def change_A(A, mask):
    """
    Outside the mask region, we want get target pixels as result. So we change these corresponding parts of A to
    I (identical matrix). So in Af = b equation, we will get target pixels for pixels that are outside the mask region.
    For this subject, we must change 4 on diagonal to 1 and -1s to 0 for each row of A matrix.

    :param A: the A matrix
    :param mask: our mask
    :return: changed A
    """
    height, width = mask.shape[:2]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask[y, x] == 0:
                k = y * width
                k += x
                A[k, k] = 1
                A[k, k + width] = 0
                A[k, k - width] = 0
                A[k, k + 1] = 0
                A[k, k - 1] = 0

    # Compress matrix A
    A = A.tocsc()
    return A


def solve_equation(source, target, mask, A):
    height, width = source.shape

    mask_flat = mask.flatten()
    source_flat = source.flatten()
    target_flat = target.flatten()

    # build b: b = As
    b = A.dot(source_flat)

    # outside the mask, we want b pixels is exactly like target image
    b[mask_flat == 0] = target_flat[mask_flat == 0]

    # solve equation Af = b
    f = spsolve(A, b)

    result_image = get_image(f, height, width)
    return result_image


def get_A_dot_s_image(A, source, height, width):
    As = A.dot(source[:, :, 0].flatten())
    As = As.reshape(height, width)

    return As


def get_image(f, height, width):
    f[f > 255] = 255
    f[f < 0] = 0
    f = f.reshape((height, width)).astype('uint8')

    return f

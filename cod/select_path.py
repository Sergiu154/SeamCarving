import sys
import numpy as np
import pdb
import cv2 as cv


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    path = [(line, col)]
    for i in range(E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = path[-1][1] + opt
        path.append((line, col))

    return path


def select_greedy_path(E):
    col = np.argmin(E[0, :])
    row = 0
    path = [(row, col)]
    row += 1
    while row < E.shape[0]:

        offset = 0
        if col == 0:
            offset = np.argmin(E[row, col:col + 2]) + 1
        elif col == E.shape[1] - 1:
            offset = np.argmin(E[row, col - 1:col + 1])

        else:
            offset = np.argmin(E[row, col - 1:col + 2])

        col += (offset - 1)

        path.append((row, col))
        row += 1
    return path


def select_dynamic_programming_path(E):
    # TODO: scrieti codul
    dp = np.copy(E)
    # dp.dtype = 'float64'
    dp[0, :] = np.copy(E[0, :])
    n, m = E.shape
    print("Linii", n, "Col", m)
    for i in range(1, n):
        for j in range(m):
            if j == 0:

                dp[i][j] += min(dp[i - 1][j], dp[i - 1][j + 1])

            elif j == m - 1:
                dp[i][j] += min(dp[i - 1][j], dp[i - 1][j - 1])

            else:
                dp[i][j] += min(dp[i - 1][j + 1], min(dp[i - 1][j], dp[i - 1][j - 1]))

    print(dp.dtype)
    curr_row = dp.shape[0] - 1
    curr_col = np.argmin(dp[-1, :])
    path = [(curr_row, curr_col)]

    print("MIN VAL", dp[curr_row][curr_col])

    while curr_row > 0:
        left = -1 if curr_col == 0 else dp[curr_row - 1][curr_col - 1]
        mid = dp[curr_row - 1][curr_col]
        right = -1 if curr_col == m - 1 else dp[curr_row - 1][curr_col + 1]
        current = dp[curr_row][curr_col]
        prev = current - E[curr_row][curr_col]

        if prev == left:
            curr_col -= 1
        elif prev == right:
            curr_col += 1

        curr_row -= 1
        path.insert(0, (curr_row, curr_col))

    return path


def select_path(E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'greedy':
        return select_greedy_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_programming_path(E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)

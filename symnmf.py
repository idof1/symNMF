import numpy as np
import symnmfmodule  # Import the C module
import sys


def printOutput(M):
    # Prints matrix output in for digits after decimal point format
    for i in range(len(M)):
        formatted_4_digits = [f"{x:.4f}" for x in M[i]]
        for j in range(len(formatted_4_digits) - 1):
            print(str(formatted_4_digits[j]) + ",", end="")
        print(str(formatted_4_digits[-1]))


def symnmf(k, goal, file_name, vectors):
    # Read the input file (assuming each line is a data point)
    X = vectors

    # Handle the goals
    if goal == 'symnmf':
        # Initialize H as per the requirements
        np.random.seed(1234)

        # Call the C module to perform symnmf
        A = symnmfmodule.sym(X.tolist())
        D = symnmfmodule.ddg(A)
        W = symnmfmodule.norm(A, D)

        m = np.mean(W)
        H = np.random.uniform(0, 2 * np.sqrt(m / k), size=(X.shape[0], k))


        final_H = symnmfmodule.symnmf(W, H.tolist(), 300, 1e-4)

        # Output final H matrix
        printOutput(final_H)

    elif goal == 'sym':
        A = symnmfmodule.sym(X.tolist())
        printOutput(A)

    elif goal == 'ddg':
        A = symnmfmodule.sym(X.tolist())
        D = symnmfmodule.ddg(A)
        printOutput(D)
    elif goal == 'norm':
        A = symnmfmodule.sym(X.tolist())
        D = symnmfmodule.ddg(A)
        W = symnmfmodule.norm(A, D)
        printOutput(W)


def read_vectors_from_txt(filename, delimiter=","):
    # Reads data from file into a numpy array
    vectors = []
    with open(filename, 'r') as f:
        for line in f:
            # Split the line based on the delimiter and convert each element to float
            vector = [float(x) for x in line.strip().split(delimiter)]
            vectors.append(vector)
    return np.array(vectors)


# Read command-line arguments
k = int(sys.argv[1])  # Number of clusters
goal = sys.argv[2]  # Task goal ('symnmf', 'sym', 'ddg', or 'norm')
file_name = sys.argv[3]  # Input file containing the data points

vector = read_vectors_from_txt(file_name)
symnmf(k, goal, file_name, vector)

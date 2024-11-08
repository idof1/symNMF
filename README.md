
# Symmetric Non-negative Matrix Factorization (SymNMF) Clustering Project

This project implements a clustering algorithm based on Symmetric Non-negative Matrix Factorization (SymNMF) and compares its performance with K-means clustering. The algorithm applies to multiple datasets, and clustering quality is evaluated using the silhouette score.

## Project Structure

- **symnmf.py**: Python interface for the project.
- **symnmf.c**: C implementation of core algorithm functions.
- **symnmfmodule.c**: Python C API wrapper to interface Python with C functions.
- **symnmf.h**: Header file containing function prototypes for `symnmfmodule.c`.
- **analysis.py**: Script to analyze clustering results and compare SymNMF with K-means.
- **setup.py**: Setup file to build the C extension for Python.
- **Makefile**: Makefile for compiling the C interface.

## Getting Started

### Prerequisites

This project requires the following:
- **Python 3**
- **NumPy**: For matrix operations.
- **scikit-learn**: For silhouette score calculation and K-means clustering.
- **C Compiler**: GCC or similar.

### Installation

1. Build the C extension with the following command:
   ```bash
   python3 setup.py build_ext --inplace
   ```

2. Compile the C components using:
   ```bash
   make
   ```

### Usage

To run the SymNMF algorithm and generate clustering results, use the following command structure:

```bash
python3 symnmf.py <k> <goal> <file_name.txt>
```

- **k**: Number of required clusters.
- **goal**: One of the following:
  - `symnmf`: Runs the complete SymNMF algorithm and outputs matrix H.
  - `sym`: Outputs the similarity matrix.
  - `ddg`: Outputs the diagonal degree matrix.
  - `norm`: Outputs the normalized similarity matrix.
- **file_name.txt**: Path to the input file containing N data points.

Example:
```bash
python3 symnmf.py 2 symnmf input_data.txt
```

### Analysis

The `analysis.py` script compares the clustering results of SymNMF and K-means. Use the following command:

```bash
python3 analysis.py <k> <file_name.txt>
```

Example:
```bash
python3 analysis.py 5 input_k5_d7.txt
```

### Outputs

Each program outputs matrices formatted to four decimal places, with each row on a new line, and values separated by commas.

## Assumptions & Notes

1. Data points are unique.
2. Outputs are formatted to four decimal places (e.g., `%.4f`).
3. Error handling:
   - Any error results in the message "An Error Has Occurred" and termination.
4. For both K-means and SymNMF, convergence criteria are set to ϵ = 1e-4 and a max iteration count of 300.

## Building & Running

To build and run the project:
1. Run `setup.py` as mentioned.
2. Ensure `make` runs without errors or warnings:
   ```bash
   make
   ```

## References

This project is based on the algorithm described in:
- Da Kuang, Chris Ding, and Haesun Park. *Symmetric Nonnegative Matrix Factorization for Graph Clustering*. SDM 2012.


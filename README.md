# SymNMF: Symmetric Non-negative Matrix Factorization for Clustering

This project is an implementation of the Symmetric Non-negative Matrix Factorization (SymNMF) clustering algorithm, developed as part of the "Software Project" course (0368-2161).

The implementation uses a hybrid **Python** and **C** approach. The core computationally intensive calculations are written in C for performance, and these functions are wrapped and exposed to a Python interface using the Python C API. The project also includes an analysis script that compares the clustering quality of this SymNMF implementation against the K-Means algorithm using the Silhouette score.

## What is SymNMF?

Symmetric Non-negative Matrix Factorization (SymNMF) is an algorithm used for graph clustering. The core idea is to:
1.  **Form a Similarity Matrix ($A$):** Represent the relationships between $n$ data points as a similarity matrix, where $a_{ij} = \exp(-\frac{||x_i - x_j||^2}{2})$ for $i \neq j$, and $a_{ii} = 0$.
2.  **Normalize the Matrix:** Compute the normalized similarity matrix $W = D^{-1/2}AD^{-1/2}$, where $D$ is the diagonal degree matrix.
3.  **Factorize:** Find a lower-dimension, non-negative matrix $H \in \mathbb{R}^{n \times k}$ (where $k$ is the number of clusters) that approximates $W$ by minimizing $||W-HH^{T}||_{F}^{2}$.
4.  **Cluster:** The resulting matrix $H$ provides an "association score" for each data point (row) to each cluster (column). Each data point is assigned to the cluster for which it has the highest association score.

## Project Architecture

This project is divided into several key components as required by the assignment:

* [cite_start]`symnmf.c` / `symnmf.h`: A C program that implements the core logic for calculating the similarity matrix (`sym`), diagonal degree matrix (`ddg`), and the normalized similarity matrix (`norm`) [cite: 52, 53, 90-92]. It also contains the functions used by the Python C API module.
* `symnmfmodule.c`: A Python C API wrapper. This file "glues" the C functions to Python, exposing them as a Python module named `symnmf`. It also includes the $H$ optimization loop, which is called by the Python script.
* `symnmf.py`: The main Python interface. It handles parsing command-line arguments, reading input files, and orchestrating the algorithm. [cite_start]It calls the C module for the `sym`, `ddg`, and `norm` goals [cite: 74-77]. [cite_start]For the full `symnmf` goal, it initializes $H$ in Python using `numpy` [cite: 68-70] and then passes $H$ and $W$ to the C module for the iterative optimization.
* `analysis.py`: A Python script that loads a dataset and compares the clustering results of SymNMF and K-Means. It reports the `silhouette_score` for both algorithms.
* `setup.py`: A standard Python setup script used to build and compile the C extension into a `.so` (shared object) file that Python can import.
* `Makefile`: A make script to build the standalone C executable `symnmf` using the required compiler flags.

## Technologies Used

* **Python 3**
* **C** (compiled with `gcc -ansi -Wall -Wextra -Werror -pedantic-errors`)
* **NumPy** (for matrix operations in Python, especially $H$ initialization)
* **Scikit-learn** (for K-Means and `silhouette_score` in `analysis.py`)

## Build Instructions

You must build both the standalone C executable and the Python C extension.

### 1. Build the C Executable

To compile the standalone `symnmf` C program, run:

```bash
make

This will create an executable file named `symnmf` in the root directory.

### 2. Build the Python C Extension

To build the `symnmf` Python module (which creates a `.so` file), run:

```bash
python3 setup.py build_ext --inplace

This command will create the `symnmf` module in your directory, allowing `symnmf.py` and `analysis.py` to import it.

## Usage

The project provides three main entry points.

### 1. Python Interface (Full SymNMF)

This is the primary script for running the full algorithm or its individual C-implemented steps.

**Syntax:**
`python3 symnmf.py <k> <goal> <file_name>`

**Arguments:**
* `k`: The number of clusters (integer).
* `goal`: The calculation to perform:
    * `symnmf`: Perform the full SymNMF algorithm and output the final matrix $H$.
    * `sym`: Calculate and output the similarity matrix $A$.
    * `ddg`: Calculate and output the diagonal degree matrix $D$.
    * `norm`: Calculate and output the normalized similarity matrix $W$.
* `file_name`: Path to the input `.txt` file containing data points.

**Example:**
```bash
python3 symnmf.py 2 symnmf input_1.txt
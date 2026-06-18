# CAG-QSAR CLI Tool

A complete, production-grade command-line QSAR (Quantitative Structure-Activity Relationship) modeling pipeline. This tool automates the process of data curation, molecular descriptor calculation, feature selection, data splitting, model building, and rigorous validation.

---

## Installation

You can install this package in Linux/WSL using three different methods, depending on your environment.

### Prerequisites
Make sure you have Python (version >= 3.8) and `pip` installed:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

### Option A: Global System-Wide Install (Recommended for Linux/WSL)
You can install the tool system-wide so that any user can run the `cagqsar` command directly:
```bash
# Clone the repository, navigate into it, and run:
sudo chmod +x install.sh
sudo ./install.sh
```
This script creates a virtual environment inside `/opt/cagqsar`, installs all prerequisites (RDKit, PyTorch CPU, XGBoost, etc.), and sets up a symbolic link under `/usr/local/bin/cagqsar`. Once installed, simply call:
```bash
cagqsar --help
```

### Option B: Local User Install
If you do not have root (`sudo`) access:
```bash
pip install --user .
```
This registers the package in your user folder. Ensure that `~/.local/bin` is added to your shell `$PATH`. You can run it using:
```bash
cagqsar --help
```

### Option C: Editable Development Mode
If you plan to modify the source code of the pipeline and want changes to reflect instantly:
```bash
pip install -e .
```

---

## Command Line Usage

After installation, run the application from any directory in your terminal:

```bash
cagqsar --data <dataset_csv> --smiles <smiles_column> --activity <activity_column> [options]
```

### Core CLI Arguments:
* `--data`: Path to the CSV dataset (Required).
* `--smiles`: Column name containing SMILES strings (Required).
* `--activity`: Column name containing activities in nM (Required).
* `--model`: Regression algorithm to train: `mlr` (MLR), `pls` (PLS), `rf` (Random Forest), `svr` (SVM), `xgb` (XGBoost), or `gnn` (Graph Neural Network) (Default: `pls`).
* `--split`: Splitting method: `random` or `pca` (Kennard-Stone PCA-distance split) (Default: `pca`).
* `--test_size`: Fraction of data allocated to the test set (Default: `0.2`).
* `--var_thresh`: Variance filter threshold for dropping constant descriptors (Default: `0.01`).
* `--corr_thresh`: Correlation threshold for collinearity filter (Default: `0.85`).
* `--y_rand_runs`: Number of Y-randomization validation loops (Default: `50`).
* `--fingerprints`: Flag to compute 2D fingerprints (Morgan/ECFP + MACCS keys) in addition to physical descriptors.
* `--out_dir`: Directory to export curated data, model reports, trained model binaries, and evaluation plots (Default: `qsar_output`).

---

## Programmatic Import in Python

Once the package is installed, you can import and use any of its internal logic (like the structure curator or descriptor calculator) in your own scripts:

```python
from cagqsar import curate_molecule, get_rdkit_descriptors

# 1. Clean a SMILES structure and remove salt fragments
clean_smiles, mol = curate_molecule("CN(C)C(=O)c1ccccc1.Cl", Chem.SaltRemover.SaltRemover())

# 2. Extract standard RDKit descriptors
descriptors = get_rdkit_descriptors(mol)
```

---

## Publishing to GitHub & PyPI

Follow these instructions to publish your code for public access.

### 1. Publishing to GitHub
Initialize the local git repository, commit the files, and push to GitHub:
```bash
# 1. Initialize repository
git init

# 2. Add files (automatically respects .gitignore)
git add .

# 3. Create initial commit
git commit -m "feat: initial release of cagqsar v1.0.0"

# 4. Set main branch name
git branch -M main

# 5. Add remote GitHub link and push
git remote add origin https://github.com/YOUR_USERNAME/cagqsar.git
git push -u origin main
```

### 2. Publishing to PyPI
To make the tool installable globally via `pip install cagqsar`, build and upload the package distributions to the Python Package Index (PyPI):

```bash
# 1. Install packaging build tools
pip install --upgrade build twine

# 2. Compile source distribution (sdist) and binary wheel (bdist_wheel)
python3 -m build

# 3. Verify build files
twine check dist/*

# 4. Upload to PyPI (requires PyPI API Token)
python3 -m twine upload dist/*
```

---

## Acknowledgments & Credits

* **Concept, Idea & Planning**: Sathish Kumar M Ponnaiya (SKM Ponnaiya).
* **Infrastructure & Support**: **Ponnaiya's Code And Genome Pvt Ltd, Madurai** (System support, server resources, internet facilities, and infrastructure).
* **AI Coding Partner**: Pair-programmed and optimized using **Antigravity**, a Google DeepMind agentic coding system.
* **Large Language Model (LLM)**: Driven by Google's **Gemini 3.5 Flash**.
* **Access Provider**: Grateful to **Jio** for enabling Gemini Premium access.
* **Test Dataset**: Sourced from the public **BindingDB database**.

*This software is open-access and free for all users under the terms of the MIT License.*

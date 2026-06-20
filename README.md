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

### Option A: Install from PyPI (Once Published)
After publishing the package to PyPI, you can create a virtual environment and install the tool globally or locally using `pip`:
```bash
# 1. Create a virtual environment
python3 -m venv qsar_env
source qsar_env/bin/activate

# 2. Install the package from PyPI
pip install cagqsar

# 3. Run the CLI tool
cagqsar --help
```

### Option B: Local Source Install (No root access required)
You can build and install the package locally from the repository folder:
```bash
# 1. Navigate to the repository directory
cd git_qsar

# 2. Create and activate a virtual environment
python3 -m venv qsar_env
source qsar_env/bin/activate

# 3. Install the package locally
pip install .

# 4. Run the CLI tool
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
* `--qsar_type`: Type of modeling: `2d` (standard descriptors + fingerprints) or `3d` (conformers, aligned grids, shape fields) (Default: `2d`).
* `--model`: Regression algorithm to train:
  - For `2d` QSAR: `mlr`, `pls`, `rf`, `svr`, `xgb`, `gnn`
  - For `3d` QSAR: `mlr`, `pls`, `rf`, `svr`, `xgb`, `cnn3d` (3D CNN), `gnn3d` (3D GNN), `pointnet` (Point Cloud Net)
* `--split`: Splitting method: `random` or `pca` (Kennard-Stone split) (Default: `pca`).
* `--test_size`: Fraction of data allocated to the test set (Default: `0.2`).
* `--var_thresh`: Variance filter threshold for dropping constant descriptors (Default: `0.01`).
* `--corr_thresh`: Correlation threshold for collinearity filter (Default: `0.85`).
* `--y_rand_runs`: Number of Y-randomization validation loops (Default: `50`).
* `--fingerprints`: Flag to compute 2D fingerprints (Morgan/ECFP + MACCS keys) - 2D QSAR only.
* `--out_dir`: Directory to export curated data, model reports, trained model binaries, and evaluation plots (Default: `qsar_output`).

---

## Programmatic Import in Python

Once the package is installed, you can import and use any of its internal logic in your own scripts:

```python
from rdkit import Chem
from cagqsar import (
    curate_molecule, 
    get_rdkit_descriptors, 
    generate_3d_conformer, 
    align_molecules_3d, 
    generate_3d_descriptors
)

# 1. Clean a SMILES structure and remove salt fragments
remover = Chem.SaltRemover.SaltRemover()
clean_smiles, mol = curate_molecule("CN(C)C(=O)c1ccccc1.Cl", remover)

# 2. Extract standard 2D RDKit descriptors
descriptors = get_rdkit_descriptors(mol)

# 3. Generate a 3D conformer and optimize geometry (for 3D QSAR)
mol_3d = generate_3d_conformer(mol)

# 4. Align molecule to a reference coordinate template
aligned_mols = align_molecules_3d([mol_3d], ref_mol=template_mol)
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

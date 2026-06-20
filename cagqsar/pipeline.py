#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdMolAlign, rdFMCS
from rdkit.Chem.SaltRemover import SaltRemover

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# XGBoost
import xgboost as xgb

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==========================================
# 1. DATA CURATION
# ==========================================

def curate_molecule(smiles, remover):
    """
    Strips salts, keeps largest organic fragment, sanitizes SMILES.
    Returns cleaned SMILES and RDKit mol object, or (None, None) if invalid.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        # Remove salt/solvent fragments
        mol_stripped = remover.StripMol(mol)
        if mol_stripped is None:
            return None, None
        
        # Pick the largest organic fragment by number of heavy atoms
        frags = Chem.GetMolFrags(mol_stripped, asMols=True)
        if not frags:
            return None, None
        
        largest_frag = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        
        # Sanitize the molecule to ensure correct valences and aromaticity
        Chem.SanitizeMol(largest_frag)
        
        # Convert back to canonical SMILES
        clean_smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=True)
        return clean_smiles, largest_frag
    except Exception:
        return None, None

def curate_dataset(df, smiles_col, activity_col):
    """
    Curates SMILES, filters compounds, cleans activity values,
    resolves duplicate molecules, and converts activities to pIC50.
    """
    print("Starting data curation...")
    remover = SaltRemover()
    cleaned_rows = []
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        raw_act = row[activity_col]
        
        # Check for empty value
        if pd.isna(raw_act) or raw_act == '':
            continue
            
        try:
            # Clean inequality symbols if present
            if isinstance(raw_act, str):
                for char in ['>', '<', '=', '~', ' ', ',']:
                    raw_act = raw_act.replace(char, '')
            act_val = float(raw_act)
            if act_val <= 0:
                continue
        except ValueError:
            continue
            
        # Curate chemical structure
        clean_smiles, mol = curate_molecule(smiles, remover)
        if mol is None:
            continue
            
        # Convert nM to M and compute -log10
        # pIC50 = -log10(act_val * 10^-9) = 9.0 - log10(act_val)
        p_val = 9.0 - np.log10(act_val)
        
        cleaned_rows.append({
            'Clean_SMILES': clean_smiles,
            'pActivity': p_val,
            'MolObject': mol
        })
        
    curated_df = pd.DataFrame(cleaned_rows)
    if curated_df.empty:
        print("Error: No valid data left after curation.")
        return pd.DataFrame()
        
    print(f"Compounds successfully curated: {len(curated_df)}")
    
    # Resolve duplicates by averaging activities
    unique_df = curated_df.groupby('Clean_SMILES').agg({
        'pActivity': 'mean',
        'MolObject': 'first'
    }).reset_index()
    
    print(f"Unique compounds remaining: {len(unique_df)}")
    return unique_df


# ==========================================
# 1.1. 3D CONFORMER GENERATION & ALIGNMENT (For 3D QSAR)
# ==========================================

def generate_3d_conformer(mol):
    """
    Generates a single low-energy 3D conformer for a molecule.
    Steps:
      1. Add explicit hydrogens.
      2. Embed molecule using ETKDGv3 (fallback to ETKDGv2/ETKDG).
      3. Optimize using MMFF94 force field (fallback to UFF).
    """
    if mol is None:
        return None
    try:
        # Add explicit hydrogens
        mol_h = Chem.AddHs(mol)
        
        # Conformer embedding parameters
        if hasattr(AllChem, 'ETKDGv3'):
            params = AllChem.ETKDGv3()
        elif hasattr(AllChem, 'ETKDGv2'):
            params = AllChem.ETKDGv2()
        else:
            params = AllChem.ETKDG()
            
        params.randomSeed = 42
        
        # Embed
        res = AllChem.EmbedMolecule(mol_h, params)
        if res < 0:
            # Fallback to standard embedding
            res = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=200)
        if res < 0:
            # Fallback with random coordinates
            params_fallback = AllChem.ETKDG()
            params_fallback.useRandomCoords = True
            params_fallback.randomSeed = 42
            res = AllChem.EmbedMolecule(mol_h, params_fallback)
            
        if res >= 0:
            try:
                # MMFF94 Optimization
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
                else:
                    AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
            except Exception:
                pass
            return mol_h
        else:
            return None
    except Exception:
        return None

def align_molecules_3d(mols, ref_mol):
    """
    Aligns a list of molecules (with 3D conformers and Hs) to the reference molecule `ref_mol` in-place.
    Uses pairwise Maximum Common Substructure (MCS) to align to reference molecule.
    """
    aligned_mols = []
    if ref_mol is None:
        return mols
        
    for mol in mols:
        if mol is None:
            aligned_mols.append(None)
            continue
        try:
            # If same chemical structure, bypass MCS
            if mol.GetNumAtoms() == ref_mol.GetNumAtoms() and Chem.MolToSmiles(mol) == Chem.MolToSmiles(ref_mol):
                aligned_mols.append(mol)
                continue
                
            # Pairwise MCS
            mcs = rdFMCS.FindMCS([mol, ref_mol], timeout=1)
            if mcs.numAtoms >= 3:
                core = Chem.MolFromSmarts(mcs.smartsString)
                if core is not None:
                    ref_matches = ref_mol.GetSubstructMatches(core)
                    mol_matches = mol.GetSubstructMatches(core)
                    if ref_matches and mol_matches:
                        atom_map = list(zip(mol_matches[0], ref_matches[0]))
                        rdMolAlign.AlignMol(mol, ref_mol, atomMap=atom_map)
            aligned_mols.append(mol)
        except Exception:
            aligned_mols.append(mol)
    return aligned_mols


# ==========================================
# 2. MOLECULAR DESCRIPTORS
# ==========================================

def get_rdkit_descriptors(mol):
    """
    Computes all standard RDKit descriptors.
    Includes Constitutional, Topological, Kier & Hall Chi/Kappa,
    Atom-Centered Fragments, Physicochemical, E-State, etc.
    """
    desc_dict = {}
    for name, func in Descriptors._descList:
        try:
            val = func(mol)
            # Handle float anomalies
            if np.isnan(val) or np.isinf(val):
                desc_dict[name] = 0.0
            else:
                desc_dict[name] = float(val)
        except Exception:
            desc_dict[name] = 0.0
    return desc_dict

def get_2d_fingerprints(mol, radius=2, n_bits=1024):
    """
    Computes 2D Morgan (ECFP) fingerprints and MACCS keys.
    """
    fp_dict = {}
    try:
        # Morgan FP
        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        for i, bit in enumerate(morgan_fp):
            fp_dict[f"Morgan_Bit_{i}"] = int(bit)
            
        # MACCS keys
        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        for i, bit in enumerate(maccs_fp):
            fp_dict[f"MACCS_Bit_{i}"] = int(bit)
    except Exception as e:
        print(f"Error computing fingerprints: {e}")
        
    return fp_dict

def generate_descriptors(df, use_fingerprints=True):
    """
    Generates descriptors for the curated dataset.
    """
    print("Calculating molecular descriptors...")
    features = []
    for mol in df['MolObject']:
        desc = get_rdkit_descriptors(mol)
        if use_fingerprints:
            fp = get_2d_fingerprints(mol)
            desc.update(fp)
        features.append(desc)
        
    features_df = pd.DataFrame(features)
    features_df.index = df.index
    
    # Fill any remaining NaNs with 0
    features_df = features_df.fillna(0.0)
    print(f"Total descriptors generated: {features_df.shape[1]}")
    return features_df


# ------------------------------------------
# 3D QSAR DESCRIPTOR ENGINES
# ------------------------------------------

def compute_gasteiger_charges(mol):
    """
    Computes Gasteiger partial charges using RDKit.
    Returns a list of charges, or 0.0 fallback for NaNs/errors.
    """
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            try:
                val = atom.GetDoubleProp('_GasteigerCharge')
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                charges.append(val)
            except Exception:
                charges.append(0.0)
        return charges
    except Exception:
        return [0.0] * mol.GetNumAtoms()

def define_grid(mols, padding=4.0, spacing=2.0):
    """
    Finds the union bounding box of all conformers in mols and generates grid coordinates.
    """
    all_coords = []
    for mol in mols:
        if mol is not None:
            try:
                conf = mol.GetConformer()
                all_coords.append(conf.GetPositions())
            except Exception:
                pass
    if not all_coords:
        raise ValueError("No conformers found to construct grid bounding box.")
    
    all_coords = np.concatenate(all_coords, axis=0)
    min_xyz = np.min(all_coords, axis=0) - padding
    max_xyz = np.max(all_coords, axis=0) + padding
    
    # Create grid steps
    xs = np.arange(min_xyz[0], max_xyz[0] + 1e-5, spacing)
    ys = np.arange(min_xyz[1], max_xyz[1] + 1e-5, spacing)
    zs = np.arange(min_xyz[2], max_xyz[2] + 1e-5, spacing)
    
    grid_shape = (len(zs), len(ys), len(xs))
    
    # Generate coordinates in (Z, Y, X) order for easy 3D reshaping
    grid_coords = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid_coords.append([x, y, z])
    grid_coords = np.array(grid_coords, dtype=np.float32)
    
    grid_bounds = {
        'min_xyz': min_xyz.tolist(),
        'max_xyz': max_xyz.tolist(),
        'spacing': spacing,
        'grid_shape': grid_shape,
        'xs': xs.tolist(),
        'ys': ys.tolist(),
        'zs': zs.tolist()
    }
    return grid_coords, grid_shape, grid_bounds

def reconstruct_grid(grid_bounds):
    xs = np.array(grid_bounds['xs'])
    ys = np.array(grid_bounds['ys'])
    zs = np.array(grid_bounds['zs'])
    grid_coords = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid_coords.append([x, y, z])
    return np.array(grid_coords, dtype=np.float32)

def compute_grid_fields_for_mol(mol, grid_coords, charges):
    """
    Computes Lennard-Jones (steric) and Coulombic (electrostatic) potentials at grid_coords.
    Caps potentials at +30.0 kcal/mol and ±30.0 kcal/mol respectively.
    """
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    
    vdw_radii = []
    vdw_eps = []
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        r, eps = 1.70, 0.070  # Default C
        if elem == 'H':
            r, eps = 1.20, 0.030
        elif elem == 'N':
            r, eps = 1.55, 0.070
        elif elem == 'O':
            r, eps = 1.52, 0.070
        elif elem == 'F':
            r, eps = 1.47, 0.075
        elif elem == 'P':
            r, eps = 1.80, 0.150
        elif elem == 'S':
            r, eps = 1.80, 0.150
        elif elem == 'Cl':
            r, eps = 1.75, 0.150
        elif elem == 'Br':
            r, eps = 1.85, 0.200
        elif elem == 'I':
            r, eps = 1.98, 0.250
        vdw_radii.append(r)
        vdw_eps.append(eps)
        
    vdw_radii = np.array(vdw_radii)
    vdw_eps = np.array(vdw_eps)
    charges = np.array(charges)
    
    r_probe = 1.70
    eps_probe = 0.070
    q_probe = 1.0
    
    diff = pos[:, np.newaxis, :] - grid_coords[np.newaxis, :, :]  # (N_atoms, N_grid, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N_atoms, N_grid)
    dists = np.maximum(dists, 0.8)  # Cap distance to avoid division by zero
    
    r0 = vdw_radii[:, np.newaxis] + r_probe
    eps_ip = np.sqrt(vdw_eps[:, np.newaxis] * eps_probe)
    ratio = r0 / dists
    ratio_6 = ratio ** 6
    ratio_12 = ratio_6 ** 2
    steric_pot = eps_ip * (ratio_12 - 2 * ratio_6)
    steric = np.sum(steric_pot, axis=0)
    
    elec_pot = (charges[:, np.newaxis] * q_probe / dists) * 332.0637
    elec = np.sum(elec_pot, axis=0)
    
    steric = np.clip(steric, -30.0, 30.0)
    elec = np.clip(elec, -30.0, 30.0)
    
    return steric, elec

def get_grid_free_3d_descriptors(mol):
    """
    Generates standard RDKit 3D descriptors: RDF, 3D-MoRSE, WHIM, GETAWAY.
    """
    desc = {}
    if mol is None:
        return desc
    try:
        rdf = rdMolDescriptors.CalcRDF(mol)
        for i, val in enumerate(rdf):
            desc[f"RDF_{i}"] = float(val) if not (np.isnan(val) or np.isinf(val)) else 0.0
    except Exception:
        pass
    try:
        morse = rdMolDescriptors.Calc3DMoRSE(mol)
        for i, val in enumerate(morse):
            desc[f"MoRSE_{i}"] = float(val) if not (np.isnan(val) or np.isinf(val)) else 0.0
    except Exception:
        pass
    try:
        whim = rdMolDescriptors.CalcWHIM(mol)
        for i, val in enumerate(whim):
            desc[f"WHIM_{i}"] = float(val) if not (np.isnan(val) or np.isinf(val)) else 0.0
    except Exception:
        pass
    try:
        getaway = rdMolDescriptors.CalcGETAWAY(mol)
        for i, val in enumerate(getaway):
            desc[f"GETAWAY_{i}"] = float(val) if not (np.isnan(val) or np.isinf(val)) else 0.0
    except Exception:
        pass
    return desc

def generate_3d_descriptors(df, grid_coords, include_grid_free=True):
    """
    Calculates 3D grid potentials and grid-free descriptors for all molecules in dataframe.
    Returns a dataframe of combined 3D descriptors, and X_grid_voxels (for 3D CNN).
    """
    print("Calculating 3D descriptors...")
    features = []
    voxel_list = []
    
    for idx, row in df.iterrows():
        mol = row['MolObject']
        charges = compute_gasteiger_charges(mol)
        steric, elec = compute_grid_fields_for_mol(mol, grid_coords, charges)
        
        desc = {}
        for i in range(len(steric)):
            desc[f"Steric_Grid_{i}"] = float(steric[i])
            desc[f"Elec_Grid_{i}"] = float(elec[i])
            
        voxel_list.append(np.stack([steric, elec], axis=0))
        
        if include_grid_free:
            grid_free = get_grid_free_3d_descriptors(mol)
            desc.update(grid_free)
            
        features.append(desc)
        
    features_df = pd.DataFrame(features)
    features_df.index = df.index
    features_df = features_df.fillna(0.0)
    
    X_grid_voxels = np.array(voxel_list, dtype=np.float32)
    
    return features_df, X_grid_voxels


# ==========================================
# 3. FEATURE SELECTION
# ==========================================

def select_features(X, y, var_thresh=0.01, corr_thresh=0.85, max_k=None):
    """
    1. Variance Filter (drops constant/near-constant features)
    2. Correlation Check (drops redundant features with r > 0.85)
    3. Lasso Feature Selection (extracts final predictive subset)
    """
    print("Running feature selection...")
    
    # 1. Variance Filter
    variances = X.var(ddof=0)
    var_cols = variances[variances > var_thresh].index.tolist()
    X_var = X[var_cols]
    print(f"Descriptors after Variance Filter (> {var_thresh}): {X_var.shape[1]}")
    
    if X_var.shape[1] <= 1:
        return X_var
        
    # 2. Correlation Check (drops one from pairs with |r| > 0.85)
    corr_matrix = X_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for col in upper.columns:
        high_corr = upper.index[upper[col] > corr_thresh].tolist()
        if high_corr:
            to_drop.append(col)
            
    X_corr = X_var.drop(columns=to_drop)
    print(f"Descriptors after Correlation Check (< {corr_thresh}): {X_corr.shape[1]}")
    
    if X_corr.shape[1] <= 1:
        return X_corr
        
    # 3. Lasso-based Feature Selection (LassoCV)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_corr)
    
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    selected_features = X_corr.columns[lasso.coef_ != 0].tolist()
    print(f"Descriptors selected by LassoCV: {len(selected_features)}")
    
    # Fallback if Lasso CV selects zero features
    if len(selected_features) == 0:
        corrs_with_y = X_corr.apply(lambda col: np.abs(stats.pearsonr(col, y)[0]))
        selected_features = corrs_with_y.nlargest(min(10, X_corr.shape[1])).index.tolist()
        print("LassoCV selected 0 features. Reverted to top features correlated with target.")
        
    # Enforce statistical constraint k < n/5 if specified
    if max_k and len(selected_features) > max_k:
        coef_imp = pd.DataFrame({
            'Feature': X_corr.columns,
            'Coef': np.abs(lasso.coef_)
        })
        # Filter to selected and sort
        coef_imp = coef_imp[coef_imp['Feature'].isin(selected_features)]
        coef_imp = coef_imp.sort_values(by='Coef', ascending=False)
        selected_features = coef_imp['Feature'].head(max_k).tolist()
        print(f"Restricted descriptors to top {max_k} to comply with n/5 rule.")
        
    return X_corr[selected_features]


# ==========================================
# 4. DATA SPLITTING
# ==========================================

def kennard_stone_split(X, test_size=0.2):
    """
    Splits data using the PCA-based Kennard-Stone algorithm.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA to cover 95% variance to speed up distance computations
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    n_samples = X_pca.shape[0]
    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test
    
    # Find the sample closest to the mean coordinate
    mean_coords = np.mean(X_pca, axis=0)
    dists_to_mean = np.sum((X_pca - mean_coords) ** 2, axis=1)
    first_choice = np.argmin(dists_to_mean)
    
    selected = [first_choice]
    remaining = list(range(n_samples))
    remaining.remove(first_choice)
    
    # Find the sample furthest from the first choice
    dists_to_first = np.sum((X_pca[remaining] - X_pca[first_choice]) ** 2, axis=1)
    second_choice = remaining[np.argmax(dists_to_first)]
    selected.append(second_choice)
    remaining.remove(second_choice)
    
    # Select training set iteratively
    while len(selected) < n_train:
        # Compute distances between remaining and selected
        # Shape: (len(remaining), len(selected))
        dists = np.sum((X_pca[remaining, np.newaxis, :] - X_pca[np.newaxis, selected, :]) ** 2, axis=2)
        min_dists = np.min(dists, axis=1)
        
        # Maximize the minimum distance
        next_choice_idx = np.argmax(min_dists)
        next_choice = remaining[next_choice_idx]
        
        selected.append(next_choice)
        remaining.remove(next_choice)
        
    return selected, remaining

def split_dataset(X, y, method='pca', test_size=0.2):
    """
    Splits data into train (80%) and test (20%) using random or PCA Kennard-Stone method.
    """
    print(f"Splitting data using method: {method.upper()}...")
    if method == 'random':
        indices = np.random.permutation(X.shape[0])
        split_idx = int(np.round(X.shape[0] * (1 - test_size)))
        train_idx = indices[:split_idx].tolist()
        test_idx = indices[split_idx:].tolist()
    else:
        train_idx, test_idx = kennard_stone_split(X, test_size)
        
    return train_idx, test_idx


# ==========================================
# 5. MODEL BUILDING & TRAINING (MLR, PLS, RF, SVR, XGB)
# ==========================================

def get_regressor(model_type, X_train):
    """
    Instantiates standard regressors.
    """
    if model_type == 'mlr':
        return LinearRegression()
    elif model_type == 'pls':
        # Optimize PLS components via GridSearchCV
        max_comps = min(10, X_train.shape[1])
        model = PLSRegression()
        gs = GridSearchCV(model, param_grid={'n_components': list(range(1, max_comps+1))}, cv=5)
        return gs
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
        gs = GridSearchCV(model, param_grid={
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }, cv=5)
        return gs
    elif model_type == 'svr':
        model = SVR(kernel='rbf')
        gs = GridSearchCV(model, param_grid={
            'C': [0.1, 1.0, 10.0, 100.0],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto']
        }, cv=5)
        return gs
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(random_state=42)
        gs = GridSearchCV(model, param_grid={
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1]
        }, cv=5)
        return gs
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ==========================================
# 6. GRAPH NEURAL NETWORK IMPLEMENTATION
# ==========================================

def get_atom_features(atom):
    """
    Generates node features for chemical graphs.
    """
    # 1. Atomic symbols (one-hot)
    symbols = [6, 7, 8, 16, 9, 17, 35, 53, 15, 1]
    symbol_onehot = [float(atom.GetAtomicNum() == s) for s in symbols]
    symbol_onehot.append(float(atom.GetAtomicNum() not in symbols))
    
    # 2. Hybridization (one-hot)
    hyb = atom.GetHybridization()
    hybs = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
    hyb_onehot = [float(hyb == h) for h in hybs]
    hyb_onehot.append(float(hyb not in hybs))
    
    # 3. Connection degree (one-hot)
    deg = atom.GetDegree()
    deg_onehot = [float(deg == d) for d in range(5)]
    deg_onehot.append(float(deg >= 5))
    
    # 4. Binary properties
    aromatic = float(atom.GetIsAromatic())
    charge = float(atom.GetFormalCharge())
    valence = float(atom.GetImplicitValence())
    
    return np.array(symbol_onehot + hyb_onehot + deg_onehot + [aromatic, charge, valence], dtype=np.float32)

def mol_to_graph(mol):
    n_atoms = mol.GetNumAtoms()
    x = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = np.array(x, dtype=np.float32)
    
    adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        
    return x, adj

if TORCH_AVAILABLE:
    class GraphDataset(Dataset):
        def __init__(self, df):
            self.data = []
            for idx, row in df.iterrows():
                try:
                    x, adj = mol_to_graph(row['MolObject'])
                    y = float(row['pActivity'])
                    self.data.append((x, adj, y))
                except Exception:
                    pass
                    
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]

    def collate_graphs(batch):
        xs = []
        adjs = []
        ys = []
        batch_indices = []
        
        for mol_idx, (x, adj, y) in enumerate(batch):
            n_nodes = x.shape[0]
            xs.append(torch.tensor(x, dtype=torch.float32))
            adjs.append(torch.tensor(adj, dtype=torch.float32))
            ys.append(y)
            batch_indices.append(torch.full((n_nodes,), mol_idx, dtype=torch.long))
            
        x_batch = torch.cat(xs, dim=0)
        batch_indices_batch = torch.cat(batch_indices, dim=0)
        y_batch = torch.tensor(ys, dtype=torch.float32)
        
        # Build block-diagonal adjacency
        total_nodes = x_batch.size(0)
        adj_batch = torch.zeros((total_nodes, total_nodes), dtype=torch.float32)
        
        offset = 0
        for adj in adjs:
            n = adj.size(0)
            adj_batch[offset:offset+n, offset:offset+n] = adj
            offset += n
            
        return x_batch, adj_batch, batch_indices_batch, y_batch

    class GCNConv(nn.Module):
        def __init__(self, in_dim, out_dim):
            super(GCNConv, self).__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            
        def forward(self, x, adj_norm):
            h = self.linear(x)
            return torch.mm(adj_norm, h)

    class GNNModel(nn.Module):
        def __init__(self, in_features=24, hidden_dim=64, num_layers=3):
            super(GNNModel, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(GCNConv(in_features, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activation = nn.ReLU()
            
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x, adj, batch_indices):
            # Compute degree normalization D^-1/2 * (A + I) * D^-1/2
            identity = torch.eye(adj.size(0), device=adj.device)
            adj_tilde = adj + identity
            
            deg = torch.sum(adj_tilde, dim=1)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            
            adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
            
            h = x
            for layer in self.layers:
                h = layer(h, adj_norm)
                h = self.activation(h)
                
            # Global mean pooling
            num_mols = int(batch_indices.max().item()) + 1
            mol_features = []
            for i in range(num_mols):
                mask = (batch_indices == i)
                if mask.sum() > 0:
                    mol_features.append(h[mask].mean(dim=0))
                else:
                    mol_features.append(torch.zeros(h.size(1), device=h.device))
                    
            pooled = torch.stack(mol_features)
            return self.regressor(pooled).squeeze(-1)

    def train_gnn(train_df, val_df=None, epochs=150, batch_size=32, lr=0.005):
        train_ds = GraphDataset(train_df)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
        
        model = GNNModel()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, adj, batch_indices, y in train_loader:
                optimizer.zero_grad()
                pred = model(x, adj, batch_indices)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * y.size(0)
            # Log progress if necessary
            
        return model

    def predict_gnn(model, df, batch_size=32):
        ds = GraphDataset(df)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
        model.eval()
        preds = []
        with torch.no_grad():
            for x, adj, batch_indices, y in loader:
                pred = model(x, adj, batch_indices)
                preds.extend(pred.cpu().numpy().tolist())
        return np.array(preds)

    # ------------------------------------------
    # 3D DEEP LEARNING MODELS (PyTorch)
    # ------------------------------------------

    class Bio3DDataset(Dataset):
        def __init__(self, df):
            self.data = []
            for idx, row in df.iterrows():
                try:
                    mol = row['MolObject']
                    if mol is not None:
                        x = [get_atom_features(atom) for atom in mol.GetAtoms()]
                        x = np.array(x, dtype=np.float32)
                        
                        n_atoms = mol.GetNumAtoms()
                        adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
                        for bond in mol.GetBonds():
                            i = bond.GetBeginAtomIdx()
                            j = bond.GetEndAtomIdx()
                            adj[i, j] = 1.0
                            adj[j, i] = 1.0
                            
                        conf = mol.GetConformer()
                        pos = conf.GetPositions()  # (n_atoms, 3)
                        
                        y = float(row['pActivity'])
                        self.data.append((x, adj, pos, y))
                except Exception:
                    pass
                    
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]

    def collate_bio3d(batch):
        xs = []
        adjs = []
        poses = []
        ys = []
        batch_indices = []
        
        for mol_idx, (x, adj, pos, y) in enumerate(batch):
            n_nodes = x.shape[0]
            xs.append(torch.tensor(x, dtype=torch.float32))
            adjs.append(torch.tensor(adj, dtype=torch.float32))
            poses.append(torch.tensor(pos, dtype=torch.float32))
            ys.append(y)
            batch_indices.append(torch.full((n_nodes,), mol_idx, dtype=torch.long))
            
        x_batch = torch.cat(xs, dim=0)
        pos_batch = torch.cat(poses, dim=0)
        batch_indices_batch = torch.cat(batch_indices, dim=0)
        y_batch = torch.tensor(ys, dtype=torch.float32)
        
        total_nodes = x_batch.size(0)
        adj_batch = torch.zeros((total_nodes, total_nodes), dtype=torch.float32)
        
        offset = 0
        for adj in adjs:
            n = adj.size(0)
            adj_batch[offset:offset+n, offset:offset+n] = adj
            offset += n
            
        return x_batch, adj_batch, pos_batch, batch_indices_batch, y_batch

    class CNN3DDataset(Dataset):
        def __init__(self, X_grid, y):
            self.X_grid = torch.tensor(X_grid, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            
        def __len__(self):
            return len(self.y)
            
        def __getitem__(self, idx):
            return self.X_grid[idx], self.y[idx]

    class Conv3DModel(nn.Module):
        def __init__(self, grid_shape):
            super(Conv3DModel, self).__init__()
            self.grid_shape = grid_shape
            self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
            
            with torch.no_grad():
                dummy_x = torch.zeros(1, 2, *grid_shape)
                x = self.conv1(dummy_x)
                if x.shape[2] >= 2 and x.shape[3] >= 2 and x.shape[4] >= 2:
                    x = self.pool1(torch.relu(x))
                else:
                    x = torch.relu(x)
                x = self.conv2(x)
                if x.shape[2] >= 2 and x.shape[3] >= 2 and x.shape[4] >= 2:
                    x = self.pool2(torch.relu(x))
                else:
                    x = torch.relu(x)
                self.flat_dim = x.numel()
                
            self.fc1 = nn.Linear(self.flat_dim, 64)
            self.fc2 = nn.Linear(64, 1)
            
        def forward(self, x):
            batch_size = x.size(0)
            x_reshaped = x.view(batch_size, 2, *self.grid_shape)
            h = self.conv1(x_reshaped)
            if h.shape[2] >= 2 and h.shape[3] >= 2 and h.shape[4] >= 2:
                h = self.pool1(torch.relu(h))
            else:
                h = torch.relu(h)
                
            h = self.conv2(h)
            if h.shape[2] >= 2 and h.shape[3] >= 2 and h.shape[4] >= 2:
                h = self.pool2(torch.relu(h))
            else:
                h = torch.relu(h)
                
            h = h.view(batch_size, -1)
            h = torch.relu(self.fc1(h))
            return self.fc2(h).squeeze(-1)

    class GNN3DConv(nn.Module):
        def __init__(self, in_dim, out_dim):
            super(GNN3DConv, self).__init__()
            self.w_self = nn.Linear(in_dim, out_dim)
            self.w_neigh = nn.Linear(in_dim, out_dim)
            
        def forward(self, x, adj, pos):
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
            dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6)  # (N, N)
            weight = adj / (dist + 1.0)
            deg = torch.sum(weight, dim=1, keepdim=True)
            weight_norm = weight / (deg + 1e-6)
            h_self = self.w_self(x)
            h_neigh = torch.mm(weight_norm, self.w_neigh(x))
            return torch.relu(h_self + h_neigh)

    class GNN3DModel(nn.Module):
        def __init__(self, in_features=24, hidden_dim=64, num_layers=3):
            super(GNN3DModel, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(GNN3DConv(in_features, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GNN3DConv(hidden_dim, hidden_dim))
                
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x, adj, pos, batch_indices):
            h = x
            for layer in self.layers:
                h = layer(h, adj, pos)
                
            num_mols = int(batch_indices.max().item()) + 1
            mol_features = []
            for i in range(num_mols):
                mask = (batch_indices == i)
                if mask.sum() > 0:
                    mol_features.append(h[mask].mean(dim=0))
                else:
                    mol_features.append(torch.zeros(h.size(1), device=h.device))
            pooled = torch.stack(mol_features)
            return self.regressor(pooled).squeeze(-1)

    class PointNetModel(nn.Module):
        def __init__(self, in_features=24, hidden_dim=64):
            super(PointNetModel, self).__init__()
            self.mlp1 = nn.Sequential(
                nn.Linear(in_features + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x, pos, batch_indices):
            feat = torch.cat([pos, x], dim=1)
            h = self.mlp1(feat)
            num_mols = int(batch_indices.max().item()) + 1
            mol_features = []
            for i in range(num_mols):
                mask = (batch_indices == i)
                if mask.sum() > 0:
                    mol_features.append(h[mask].max(dim=0)[0])
                else:
                    mol_features.append(torch.zeros(h.size(1), device=h.device))
            pooled = torch.stack(mol_features)
            return self.mlp2(pooled).squeeze(-1)

    def train_3d_dl(model_type, train_df, grid_shape=None, X_grid_voxels=None, epochs=100, batch_size=32, lr=0.005):
        if model_type == 'cnn3d':
            y = train_df['pActivity'].values
            dataset = CNN3DDataset(X_grid_voxels, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model = Conv3DModel(grid_shape)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.MSELoss()
            model.train()
            for epoch in range(epochs):
                for x, y_batch in loader:
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
            return model
        elif model_type in ['gnn3d', 'pointnet']:
            dataset = Bio3DDataset(train_df)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_bio3d)
            if model_type == 'gnn3d':
                model = GNN3DModel()
            else:
                model = PointNetModel()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.MSELoss()
            model.train()
            for epoch in range(epochs):
                for x, adj, pos, batch_indices, y_batch in loader:
                    optimizer.zero_grad()
                    if model_type == 'gnn3d':
                        pred = model(x, adj, pos, batch_indices)
                    else:
                        pred = model(x, pos, batch_indices)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
            return model
        else:
            raise ValueError(f"Unknown DL model type: {model_type}")

    def predict_3d_dl(model_type, model, df, X_grid_voxels=None, batch_size=32):
        model.eval()
        if model_type == 'cnn3d':
            y = np.zeros(len(df))
            dataset = CNN3DDataset(X_grid_voxels, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for x, _ in loader:
                    pred = model(x)
                    preds.extend(pred.cpu().numpy().tolist())
            return np.array(preds)
        elif model_type in ['gnn3d', 'pointnet']:
            dataset = Bio3DDataset(df)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_bio3d)
            preds = []
            with torch.no_grad():
                for x, adj, pos, batch_indices, _ in loader:
                    if model_type == 'gnn3d':
                        pred = model(x, adj, pos, batch_indices)
                    else:
                        pred = model(x, pos, batch_indices)
                    preds.extend(pred.cpu().numpy().tolist())
            return np.array(preds)
        else:
            raise ValueError(f"Unknown DL model type: {model_type}")
else:
    def train_3d_dl(*args, **kwargs):
        raise ImportError("PyTorch is required for 3D deep learning models. Please install it using: pip install cagqsar[torch] or pip install torch")

    def predict_3d_dl(*args, **kwargs):
        raise ImportError("PyTorch is required for 3D deep learning models. Please install it using: pip install cagqsar[torch] or pip install torch")


# ==========================================
# 7. MODEL EVALUATION & STATISTICS
# ==========================================

def calculate_q2(model_type, X_train, y_train, train_df=None, grid_shape=None, X_grid_voxels_train=None):
    """
    Computes 5-fold cross-validated Q^2.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    if model_type == 'gnn':
        if not TORCH_AVAILABLE:
            return 0.0
        cv_preds = np.zeros(len(train_df))
        for train_idx, val_idx in kf.split(train_df):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            
            net = train_gnn(fold_train, epochs=80, batch_size=32, lr=0.005)
            preds = predict_gnn(net, fold_val)
            cv_preds[val_idx] = preds
        return r2_score(y_train, cv_preds)
    elif model_type in ['cnn3d', 'gnn3d', 'pointnet']:
        if not TORCH_AVAILABLE:
            return 0.0
        cv_preds = np.zeros(len(train_df))
        for train_idx, val_idx in kf.split(train_df):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            if model_type == 'cnn3d':
                voxels_fold_train = X_grid_voxels_train[train_idx]
                voxels_fold_val = X_grid_voxels_train[val_idx]
                net = train_3d_dl('cnn3d', fold_train, grid_shape=grid_shape, X_grid_voxels=voxels_fold_train, epochs=30, batch_size=32, lr=0.005)
                preds = predict_3d_dl('cnn3d', net, fold_val, X_grid_voxels=voxels_fold_val)
            else:
                net = train_3d_dl(model_type, fold_train, epochs=30, batch_size=32, lr=0.005)
                preds = predict_3d_dl(model_type, net, fold_val)
            cv_preds[val_idx] = preds
        return r2_score(y_train, cv_preds)
    else:
        # Standard models
        reg = get_regressor(model_type, X_train)
        cv_preds = cross_val_predict(reg, X_train, y_train, cv=kf)
        return r2_score(y_train, cv_preds)

def calculate_y_randomization(model_type, X_train, y_train, train_df=None, n_runs=50, grid_shape=None, X_grid_voxels_train=None):
    """
    Performs Y-randomization test to evaluate chance correlation.
    """
    print(f"Performing Y-randomization with {n_runs} runs...")
    ran_r2s = []
    ran_q2s = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for run in range(n_runs):
        y_shuffled = np.random.permutation(y_train)
        
        if model_type == 'gnn':
            df_shuffled = train_df.copy()
            df_shuffled['pActivity'] = y_shuffled
            net = train_gnn(df_shuffled, epochs=40, batch_size=32, lr=0.005)
            preds = predict_gnn(net, df_shuffled)
            r2 = r2_score(y_shuffled, preds)
            
            cv_preds = np.zeros(len(df_shuffled))
            for train_idx, val_idx in kf.split(df_shuffled):
                fold_train = df_shuffled.iloc[train_idx]
                fold_val = df_shuffled.iloc[val_idx]
                fold_net = train_gnn(fold_train, epochs=30, batch_size=32, lr=0.005)
                cv_preds[val_idx] = predict_gnn(fold_net, fold_val)
            q2 = r2_score(y_shuffled, cv_preds)
        elif model_type in ['cnn3d', 'gnn3d', 'pointnet']:
            df_shuffled = train_df.copy()
            df_shuffled['pActivity'] = y_shuffled
            if model_type == 'cnn3d':
                net = train_3d_dl('cnn3d', df_shuffled, grid_shape=grid_shape, X_grid_voxels=X_grid_voxels_train, epochs=20, batch_size=32, lr=0.005)
                preds = predict_3d_dl('cnn3d', net, df_shuffled, X_grid_voxels=X_grid_voxels_train)
            else:
                net = train_3d_dl(model_type, df_shuffled, epochs=20, batch_size=32, lr=0.005)
                preds = predict_3d_dl(model_type, net, df_shuffled)
            r2 = r2_score(y_shuffled, preds)
            
            cv_preds = np.zeros(len(df_shuffled))
            for train_idx, val_idx in kf.split(df_shuffled):
                fold_train = df_shuffled.iloc[train_idx]
                fold_val = df_shuffled.iloc[val_idx]
                if model_type == 'cnn3d':
                    voxels_fold_train = X_grid_voxels_train[train_idx]
                    voxels_fold_val = X_grid_voxels_train[val_idx]
                    fold_net = train_3d_dl('cnn3d', fold_train, grid_shape=grid_shape, X_grid_voxels=voxels_fold_train, epochs=15, batch_size=32, lr=0.005)
                    cv_preds[val_idx] = predict_3d_dl('cnn3d', fold_net, fold_val, X_grid_voxels=voxels_fold_val)
                else:
                    fold_net = train_3d_dl(model_type, fold_train, epochs=15, batch_size=32, lr=0.005)
                    cv_preds[val_idx] = predict_3d_dl(model_type, fold_net, fold_val)
            q2 = r2_score(y_shuffled, cv_preds)
        else:
            reg = get_regressor(model_type, X_train)
            if hasattr(reg, 'estimator'):  # GridSearchCV wrapper
                reg.fit(X_train, y_train)
                best_model = reg.best_estimator_
                best_model.fit(X_train, y_shuffled)
                preds = best_model.predict(X_train)
                r2 = r2_score(y_shuffled, preds)
                cv_preds = cross_val_predict(best_model, X_train, y_shuffled, cv=kf)
                q2 = r2_score(y_shuffled, cv_preds)
            else:
                reg.fit(X_train, y_shuffled)
                preds = reg.predict(X_train)
                r2 = r2_score(y_shuffled, preds)
                cv_preds = cross_val_predict(reg, X_train, y_shuffled, cv=kf)
                q2 = r2_score(y_shuffled, cv_preds)
                
        ran_r2s.append(r2)
        ran_q2s.append(q2)
        
    best_ran_r2 = max(ran_r2s)
    best_ran_q2 = max(ran_q2s)
    
    return ran_r2s, ran_q2s, best_ran_r2, best_ran_q2

def evaluate_qsar_model(model_type, X_train, y_train, X_test, y_test, train_df=None, test_df=None, k=0, y_rand_runs=50, grid_shape=None, X_grid_voxels_train=None, X_grid_voxels_test=None):
    """
    Computes all statistical metrics required for QSAR model evaluation.
    """
    print("Evaluating model and computing statistical parameters...")
    n_train = len(y_train)
    n_test = len(y_test)
    df_val = n_train - k - 1
    
    # 1. Fit final model and predict
    if model_type == 'gnn':
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Install PyTorch to run GNN model.")
        net = train_gnn(train_df, epochs=150, batch_size=32, lr=0.005)
        train_pred = predict_gnn(net, train_df)
        test_pred = predict_gnn(net, test_df)
        model_object = net
    elif model_type in ['cnn3d', 'gnn3d', 'pointnet']:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Install PyTorch to run Deep Learning model.")
        net = train_3d_dl(model_type, train_df, grid_shape=grid_shape, X_grid_voxels=X_grid_voxels_train, epochs=100, batch_size=32, lr=0.005)
        train_pred = predict_3d_dl(model_type, net, train_df, X_grid_voxels=X_grid_voxels_train)
        test_pred = predict_3d_dl(model_type, net, test_df, X_grid_voxels=X_grid_voxels_test)
        model_object = net
    else:
        reg = get_regressor(model_type, X_train)
        reg.fit(X_train, y_train)
        train_pred = reg.predict(X_train)
        test_pred = reg.predict(X_test)
        model_object = reg
        
    # 2. Compute training R2 and Q2
    r2 = r2_score(y_train, train_pred)
    q2 = calculate_q2(model_type, X_train, y_train, train_df=train_df, grid_shape=grid_shape, X_grid_voxels_train=X_grid_voxels_train)
    
    # 3. Compute external test set metrics (Tropsha's pred_R2)
    # pred_R2 = 1 - sum(y_test - pred_test)^2 / sum(y_test - mean_train)^2
    ss_res = np.sum((y_test - test_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_train)) ** 2)
    pred_r2 = 1.0 - (ss_res / ss_tot)
    
    # 4. Standard Error of Estimate (SEE)
    see = np.sqrt(np.sum((y_train - train_pred) ** 2) / df_val)
    
    # 5. F-test statistic and probability
    # F = (R2 / k) / ((1 - R2) / df)
    if k > 0 and (1.0 - r2) > 0:
        f_stat = (r2 / k) / ((1.0 - r2) / df_val)
        f_prob = stats.f.sf(f_stat, k, df_val)
    else:
        f_stat = 0.0
        f_prob = 1.0
        
    # 6. Y-randomization
    ran_r2s, ran_q2s, best_ran_r2, best_ran_q2 = calculate_y_randomization(
        model_type, X_train, y_train, train_df=train_df, n_runs=y_rand_runs, grid_shape=grid_shape, X_grid_voxels_train=X_grid_voxels_train
    )
    
    # Z-score of real R2 compared to randomized R2s
    mean_ran_r2 = np.mean(ran_r2s)
    std_ran_r2 = np.std(ran_r2s)
    zscore = (r2 - mean_ran_r2) / std_ran_r2 if std_ran_r2 > 0 else 0.0
    
    # alpha significance: proportion of random runs with R2 >= actual R2
    alpha = np.sum(np.array(ran_r2s) >= r2) / len(ran_r2s)
    
    metrics = {
        'n_molecules': n_train,
        'k_descriptors': k,
        'df': df_val,
        'r2': r2,
        'q2': q2,
        'pred_r2': pred_r2,
        'SEE': see,
        'F-test': f_stat,
        'F_prob': f_prob,
        'Zscore': zscore,
        'best_ran_q2': best_ran_q2,
        'best_ran_r2': best_ran_r2,
        'alpha': alpha
    }
    
    return metrics, train_pred, test_pred, model_object


# ==========================================
# 8. VISUALIZATION & OUTPUT
# ==========================================

def plot_qsar_results(y_train, train_pred, y_test, test_pred, model_name, output_dir):
    """
    Plots predicted vs experimental values.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_train, train_pred, color='#3f51b5', alpha=0.7, label='Training Set')
    plt.scatter(y_test, test_pred, color='#ff4081', alpha=0.7, label='Test Set')
    
    all_vals = np.concatenate([y_train, train_pred, y_test, test_pred])
    min_val, max_val = all_vals.min() - 0.5, all_vals.max() + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.xlabel('Experimental pIC50', fontsize=12)
    plt.ylabel('Predicted pIC50', fontsize=12)
    plt.title(f'QSAR Model Performance: {model_name.upper()}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = os.path.join(output_dir, f'{model_name}_pred_vs_exp.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def run_prediction(predict_csv, model_path, smiles_col, out_dir):
    print(f"Loading trained QSAR model from: {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading model file: {e}")
        sys.exit(1)
        
    if not isinstance(model_data, dict) or 'model_object' not in model_data:
        print("Error: The selected model file does not contain a valid QSAR pipeline state.")
        print("Note: Models must be trained with cagqsar to support prediction mode.")
        sys.exit(1)
        
    model_object = model_data['model_object']
    model_name = model_data['model_name']
    selected_features = model_data.get('selected_features', None)
    use_fingerprints = model_data.get('use_fingerprints', False)
    
    print(f"Successfully loaded {model_name.upper()} model pipeline.")
    
    print(f"Reading prediction compounds from: {predict_csv}...")
    try:
        df = pd.read_csv(predict_csv)
    except Exception as e:
        print(f"Error reading prediction file: {e}")
        sys.exit(1)
        
    if smiles_col not in df.columns:
        print(f"Error: SMILES column '{smiles_col}' not found in the prediction CSV file.")
        sys.exit(1)
        
    # Curate chemical structures
    remover = SaltRemover()
    cleaned_smiles_list = []
    mol_objects = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        clean_smiles, mol = curate_molecule(smiles, remover)
        if mol is not None:
            cleaned_smiles_list.append(clean_smiles)
            mol_objects.append(mol)
            valid_indices.append(idx)
            
    if not mol_objects:
        print("Error: No valid chemical structures found in the prediction file after curation.")
        sys.exit(1)
        
    print(f"Curated {len(mol_objects)} / {len(df)} compounds successfully.")
    
    # Subset dataframe to valid rows
    pred_df = df.iloc[valid_indices].copy()
    pred_df['Clean_SMILES'] = cleaned_smiles_list
    pred_df['MolObject'] = mol_objects
    
    # Perform prediction based on model type
    qsar_type = model_data.get('qsar_type', '2d')
    if qsar_type == '3d':
        print("Generating 3D conformers for prediction compounds...")
        conformers = []
        valid_idx_3d = []
        for idx, row in pred_df.iterrows():
            mol_3d = generate_3d_conformer(row['MolObject'])
            if mol_3d is not None:
                conformers.append(mol_3d)
                valid_idx_3d.append(idx)
        
        if not conformers:
            print("Error: Conformer generation failed for all prediction compounds.")
            sys.exit(1)
            
        print(f"Successfully generated 3D conformers for {len(conformers)} / {len(valid_indices)} prediction compounds.")
        pred_df = df.loc[valid_idx_3d].copy()
        pred_df['Clean_SMILES'] = [cleaned_smiles_list[valid_indices.index(idx)] for idx in valid_idx_3d]
        pred_df['MolObject'] = conformers
        
        ref_mol = model_data.get('ref_mol', None)
        if ref_mol is not None:
            print("Aligning 3D conformers to reference template molecule...")
            pred_df['MolObject'] = align_molecules_3d(pred_df['MolObject'].tolist(), ref_mol)
            
        if model_name in ['cnn3d', 'gnn3d', 'pointnet']:
            if not TORCH_AVAILABLE:
                print(f"Error: PyTorch is required to run predictions with {model_name.upper()} model.")
                sys.exit(1)
            if model_name == 'cnn3d':
                grid_bounds = model_data.get('grid_bounds', None)
                grid_coords = reconstruct_grid(grid_bounds)
                _, X_grid_voxels = generate_3d_descriptors(pred_df, grid_coords, include_grid_free=False)
                preds = predict_3d_dl('cnn3d', model_object, pred_df, X_grid_voxels=X_grid_voxels)
            else:
                pred_df['pActivity'] = 0.0
                preds = predict_3d_dl(model_name, model_object, pred_df)
        else:
            grid_bounds = model_data.get('grid_bounds', None)
            grid_coords = reconstruct_grid(grid_bounds)
            include_grid_free = model_data.get('include_grid_free', False)
            desc_df, _ = generate_3d_descriptors(pred_df, grid_coords, include_grid_free=include_grid_free)
            
            missing_features = [f for f in selected_features if f not in desc_df.columns]
            if missing_features:
                for f in missing_features:
                    desc_df[f] = 0.0
            X = desc_df[selected_features]
            
            preds = model_object.predict(X)
            if len(preds.shape) > 1 and preds.shape[1] == 1:
                preds = preds.squeeze(-1)
    else:
        # 2D QSAR predictions
        if model_name != 'gnn':
            print("Generating molecular descriptors for predictions...")
            desc_df = generate_descriptors(pred_df, use_fingerprints=use_fingerprints)
            
            missing_features = [f for f in selected_features if f not in desc_df.columns]
            if missing_features:
                for f in missing_features:
                    desc_df[f] = 0.0
            X = desc_df[selected_features]
            
            preds = model_object.predict(X)
            if len(preds.shape) > 1 and preds.shape[1] == 1:
                preds = preds.squeeze(-1)
        else:
            if not TORCH_AVAILABLE:
                print("Error: PyTorch is required to run predictions with the GNN model.")
                sys.exit(1)
            print("Running GNN forward pass predictions...")
            pred_df['pActivity'] = 0.0  # Dummy label needed for loader
            preds = predict_gnn(model_object, pred_df)
            
    # Append predictions
    pred_df['Predicted_pIC50'] = preds
    pred_df['Predicted_IC50_nM'] = 10 ** (9.0 - preds)
    
    # Remove temporary MolObject column
    pred_df = pred_df.drop(columns=['MolObject'])
    
    # Save output
    os.makedirs(out_dir, exist_ok=True)
    out_filename = os.path.basename(predict_csv).replace('.csv', '_predicted.csv')
    out_path = os.path.join(out_dir, out_filename)
    pred_df.to_csv(out_path, index=False)
    
    print("\n" + "="*50)
    print("               PREDICTION REPORT")
    print("="*50)
    print(f"Input File:          {predict_csv}")
    print(f"Model Algorithm:     {model_name.upper()}")
    print(f"Output File Saved:   {out_path}")
    print("-"*50)
    print("Sample Predictions:")
    for i, (idx, row) in enumerate(pred_df.head(5).iterrows()):
        print(f"  {i+1}. SMILES: {row['Clean_SMILES'][:40]}... -> pIC50: {row['Predicted_pIC50']:.4f} ({row['Predicted_IC50_nM']:.2f} nM)")
    print("="*50 + "\n")


# ==========================================
# CLI MAIN ENTRY
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Complete QSAR Pipeline Command Line Software")
    # Training arguments
    parser.add_argument('--data', type=str, help="Path to raw CSV dataset")
    parser.add_argument('--smiles', type=str, help="Column name containing SMILES strings")
    parser.add_argument('--activity', type=str, help="Column name containing activity values (nM)")
    parser.add_argument('--qsar_type', type=str, default='2d',
                        choices=['2d', '3d'],
                        help="QSAR type: 2d (standard chemical descriptors) or 3d (conformers, grid, shapes)")
    parser.add_argument('--model', type=str, default='pls', 
                        choices=['mlr', 'pls', 'rf', 'svr', 'xgb', 'gnn', 'cnn3d', 'gnn3d', 'pointnet'], 
                        help="Model selection: mlr, pls, rf, svr, xgb, gnn, cnn3d, gnn3d, pointnet")
    parser.add_argument('--split', type=str, default='pca', 
                        choices=['random', 'pca'], 
                        help="Data splitting method: random, pca (Kennard-Stone)")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of test dataset")
    parser.add_argument('--var_thresh', type=float, default=0.01, help="Variance filter threshold")
    parser.add_argument('--corr_thresh', type=float, default=0.85, help="Collinearity correlation limit")
    parser.add_argument('--y_rand_runs', type=int, default=50, help="Number of Y-randomization iterations")
    parser.add_argument('--fingerprints', action='store_true', help="Use 2D fingerprints (Morgan + MACCS) - 2D QSAR only")
    parser.add_argument('--out_dir', type=str, default='qsar_output', help="Directory to save output files and plots")
    
    # Prediction arguments
    parser.add_argument('--predict', type=str, help="Path to CSV file containing new compounds to predict")
    parser.add_argument('--model_path', type=str, help="Path to the trained QSAR model (.pkl file)")
    
    args = parser.parse_args()
    
    # Handle Prediction Mode
    if args.predict:
        if not args.model_path:
            print("Error: --model_path is required when running in prediction mode.")
            sys.exit(1)
        if not args.smiles:
            print("Error: --smiles column name is required to parse the prediction CSV.")
            sys.exit(1)
        run_prediction(args.predict, args.model_path, args.smiles, args.out_dir)
        sys.exit(0)
        
    # Handle Training Mode (Default)
    if not args.data or not args.smiles or not args.activity:
        print("Error: For training a new model, --data, --smiles, and --activity are all required.")
        print("To predict new compounds instead, use: cagqsar --predict <csv_file> --model_path <pkl_file> --smiles <column>")
        sys.exit(1)
        
    # Validate qsar_type and model compatibility
    if args.qsar_type == '2d':
        if args.model in ['cnn3d', 'gnn3d', 'pointnet']:
            print(f"Error: Model '{args.model}' is a 3D model, but --qsar_type is set to '2d'.")
            sys.exit(1)
    elif args.qsar_type == '3d':
        if args.model == 'gnn':
            print("Error: Model 'gnn' is a 2D graph model. For 3D graph QSAR, please select 'gnn3d'.")
            sys.exit(1)
            
    # Check PyTorch dependency for DL models
    if args.model in ['gnn', 'cnn3d', 'gnn3d', 'pointnet'] and not TORCH_AVAILABLE:
        print(f"Error: PyTorch is not installed. Model '{args.model}' cannot be run. Please install PyTorch first.")
        sys.exit(1)
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Step 1: Data Curation
    try:
        raw_df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        sys.exit(1)
        
    if args.smiles not in raw_df.columns or args.activity not in raw_df.columns:
        print(f"Error: Specified columns '{args.smiles}' or '{args.activity}' do not exist in the dataset.")
        sys.exit(1)
        
    curated_df = curate_dataset(raw_df, args.smiles, args.activity)
    if curated_df.empty:
        sys.exit(1)
        
    # Save cleaned data
    cleaned_csv_path = os.path.join(args.out_dir, "cleaned_dataset.csv")
    curated_df.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned curated dataset saved to: {cleaned_csv_path}")
    
    # Step 2: 2D or 3D QSAR Workflow Configuration
    ref_mol = None
    grid_bounds = None
    X_grid_voxels_train = None
    X_grid_voxels_test = None
    grid_shape = None
    include_grid_free = False
    X_selected = None
    
    if args.qsar_type == '3d':
        # Step 2.1: 3D Conformer Generation
        print("Generating 3D conformers for curated structures...")
        conformers = []
        valid_indices_3d = []
        for idx, row in curated_df.iterrows():
            mol_3d = generate_3d_conformer(row['MolObject'])
            if mol_3d is not None:
                conformers.append(mol_3d)
                valid_indices_3d.append(idx)
        
        if not conformers:
            print("Error: Conformer generation failed for all curated molecules.")
            sys.exit(1)
            
        print(f"Successfully generated 3D conformers for {len(conformers)} / {len(curated_df)} compounds.")
        curated_df = curated_df.iloc[valid_indices_3d].copy()
        curated_df['MolObject'] = conformers
        
        # Step 2.2: 3D Molecular Scaffold Alignment
        # Find template: the molecule with the highest pActivity
        ref_idx = curated_df['pActivity'].idxmax()
        ref_mol = curated_df.loc[ref_idx, 'MolObject']
        ref_smiles = curated_df.loc[ref_idx, 'Clean_SMILES']
        print(f"Selected reference molecule (highest activity) for 3D alignment: {ref_smiles}")
        
        aligned_mols = align_molecules_3d(curated_df['MolObject'].tolist(), ref_mol)
        curated_df['MolObject'] = aligned_mols
        
        # Step 2.3: 3D Descriptor Calculation & Splitting
        if args.model not in ['gnn3d', 'pointnet']:
            # Define 3D Grid
            grid_coords, grid_shape, grid_bounds = define_grid(curated_df['MolObject'].tolist())
            print(f"Generated 3D grid of shape {grid_shape} containing {len(grid_coords)} points.")
            
            # Compute 3D descriptors
            include_grid_free = args.model in ['mlr', 'rf', 'svr', 'xgb']
            X, X_grid_voxels = generate_3d_descriptors(curated_df, grid_coords, include_grid_free=include_grid_free)
            y = curated_df['pActivity'].values
            
            if args.model == 'cnn3d':
                # Splitting dataset using PCA on grid features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                train_idx, test_idx = split_dataset(pd.DataFrame(X_scaled), y, method=args.split, test_size=args.test_size)
                
                X_train, X_test = None, None
                y_train, y_test = y[train_idx], y[test_idx]
                train_df_sub = curated_df.iloc[train_idx]
                test_df_sub = curated_df.iloc[test_idx]
                X_grid_voxels_train = X_grid_voxels[train_idx]
                X_grid_voxels_test = X_grid_voxels[test_idx]
                k_count = 0
            else:
                # Classical models: run feature selection
                n_est_train = int(len(y) * (1.0 - args.test_size))
                max_k = int(n_est_train / 5)
                X_selected = select_features(X, y, var_thresh=args.var_thresh, corr_thresh=args.corr_thresh, max_k=max_k)
                
                if X_selected.shape[1] == 0:
                    print("Error: All descriptors dropped by feature selection filters.")
                    sys.exit(1)
                    
                train_idx, test_idx = split_dataset(X_selected, y, method=args.split, test_size=args.test_size)
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                train_df_sub, test_df_sub = None, None
                k_count = X_train.shape[1]
        else:
            # GNN3D / PointNet
            y = curated_df['pActivity'].values
            train_idx, test_idx = split_dataset(curated_df[['Clean_SMILES']], y, method='random', test_size=args.test_size)
            train_df_sub = curated_df.iloc[train_idx]
            test_df_sub = curated_df.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test = None, None
            k_count = 0
    else:
        # 2D QSAR Flow
        if args.model != 'gnn':
            X = generate_descriptors(curated_df, use_fingerprints=args.fingerprints)
            y = curated_df['pActivity'].values
            
            n_est_train = int(len(y) * (1.0 - args.test_size))
            max_k = int(n_est_train / 5)
            X_selected = select_features(X, y, var_thresh=args.var_thresh, corr_thresh=args.corr_thresh, max_k=max_k)
            
            if X_selected.shape[1] == 0:
                print("Error: All descriptors dropped by feature selection filters.")
                sys.exit(1)
                
            train_idx, test_idx = split_dataset(X_selected, y, method=args.split, test_size=args.test_size)
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_df_sub, test_df_sub = None, None
            k_count = X_train.shape[1]
        else:
            y = curated_df['pActivity'].values
            train_idx, test_idx = split_dataset(curated_df[['Clean_SMILES']], y, method='random', test_size=args.test_size)
            train_df_sub = curated_df.iloc[train_idx]
            test_df_sub = curated_df.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test = None, None
            k_count = 0
            
    # Check sample size limit
    if len(y_train) <= 20:
        print(f"Warning: Training set has only {len(y_train)} molecules. QSAR models statistically require > 20 molecules.")
        
    # Step 5 & 6: Model Building & Evaluation
    metrics, train_pred, test_pred, model_object = evaluate_qsar_model(
        args.model, X_train, y_train, X_test, y_test,
        train_df=train_df_sub, test_df=test_df_sub, k=k_count, y_rand_runs=args.y_rand_runs,
        grid_shape=grid_shape, X_grid_voxels_train=X_grid_voxels_train, X_grid_voxels_test=X_grid_voxels_test
    )
    
    # Save the trained model file (dictionary state)
    model_path = os.path.join(args.out_dir, f"qsar_model_{args.model}.pkl")
    model_data = {
        'model_object': model_object,
        'model_name': args.model,
        'qsar_type': args.qsar_type,
        'ref_mol': ref_mol if args.qsar_type == '3d' else None,
        'grid_bounds': grid_bounds if (args.qsar_type == '3d' and args.model in ['mlr', 'pls', 'rf', 'svr', 'xgb', 'cnn3d']) else None,
        'selected_features': X_selected.columns.tolist() if X_selected is not None else None,
        'use_fingerprints': args.fingerprints if args.qsar_type == '2d' else False,
        'include_grid_free': include_grid_free if args.qsar_type == '3d' else False
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Trained model pipeline state saved to: {model_path}")
    
    # Step 7: Plot predicted vs experimental scatter plots
    plot_qsar_results(y_train, train_pred, y_test, test_pred, args.model, args.out_dir)
    
    # Print beautiful results report
    print("\n" + "="*50)
    print("                QSAR MODEL REPORT")
    print("="*50)
    print(f"Dataset Size:           {len(curated_df)} compounds")
    print(f"Model Algorithm:        {args.model.upper()}")
    print(f"Data Splitting:         {args.split.upper()} (Train: {len(y_train)}, Test: {len(y_test)})")
    print(f"Descriptors in Model (k): {metrics['k_descriptors']}")
    print(f"Degree of Freedom (df):   {metrics['df']}")
    print("-"*50)
    print(f"Training R2:            {metrics['r2']:.4f}  (Target: > 0.7)")
    print(f"Cross-Validated Q2:      {metrics['q2']:.4f}  (Target: > 0.5)")
    print(f"External Test Pred_R2:  {metrics['pred_r2']:.4f}  (Target: > 0.5)")
    print(f"Standard Error (SEE):   {metrics['SEE']:.4f}")
    print(f"F-Test Statistic:       {metrics['F-test']:.4f}")
    print(f"Alpha Error Prob (F):   {metrics['F_prob']:.4e}")
    print("-"*50)
    print("Y-RANDOMIZATION VALIDATION:")
    print(f"Best Randomized R2:     {metrics['best_ran_r2']:.4f}")
    print(f"Best Randomized Q2:     {metrics['best_ran_q2']:.4f}")
    print(f"Z-score:                {metrics['Zscore']:.4f}  (Higher is better)")
    print(f"Alpha Significance:     {metrics['alpha']:.4f}  (Target: < 0.01)")
    print("="*50 + "\n")
    
    # Write report file
    report_path = os.path.join(args.out_dir, f"qsar_report_{args.model}.txt")
    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("                QSAR MODEL REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset Size:           {len(curated_df)} compounds\n")
        f.write(f"Model Algorithm:        {args.model.upper()}\n")
        f.write(f"Data Splitting:         {args.split.upper()} (Train: {len(y_train)}, Test: {len(y_test)})\n")
        f.write(f"Descriptors in Model (k): {metrics['k_descriptors']}\n")
        f.write(f"Degree of Freedom (df):   {metrics['df']}\n")
        f.write("-"*50 + "\n")
        f.write(f"Training R2:            {metrics['r2']:.4f}\n")
        f.write(f"Cross-Validated Q2:      {metrics['q2']:.4f}\n")
        f.write(f"External Test Pred_R2:  {metrics['pred_r2']:.4f}\n")
        f.write(f"Standard Error (SEE):   {metrics['SEE']:.4f}\n")
        f.write(f"F-Test Statistic:       {metrics['F-test']:.4f}\n")
        f.write(f"Alpha Error Prob (F):   {metrics['F_prob']:.4e}\n")
        f.write("-"*50 + "\n")
        f.write("Y-RANDOMIZATION VALIDATION:\n")
        f.write(f"Best Randomized R2:     {metrics['best_ran_r2']:.4f}\n")
        f.write(f"Best Randomized Q2:     {metrics['best_ran_q2']:.4f}\n")
        f.write(f"Z-score:                {metrics['Zscore']:.4f}\n")
        f.write(f"Alpha Significance:     {metrics['alpha']:.4f}\n")
        f.write("="*50 + "\n")
    print(f"Report file successfully saved to: {report_path}")

if __name__ == '__main__':
    main()

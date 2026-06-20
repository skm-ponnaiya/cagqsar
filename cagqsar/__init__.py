# CAG-QSAR Package Initialization
# Exposes key pipeline functions for programmatic use in Python scripts

from .pipeline import (
    curate_molecule,
    curate_dataset,
    get_rdkit_descriptors,
    get_2d_fingerprints,
    generate_descriptors,
    select_features,
    kennard_stone_split,
    split_dataset,
    evaluate_qsar_model,
    generate_3d_conformer,
    align_molecules_3d,
    generate_3d_descriptors,
    train_3d_dl,
    predict_3d_dl,
    main
)

__version__ = "1.2.0"
__author__ = "Sathish Kumar M Ponnaiya"

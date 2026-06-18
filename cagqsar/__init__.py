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
    main
)

__version__ = "1.0.0"
__author__ = "Sathish Kumar M Ponnaiya"

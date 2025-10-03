import numpy as np

from molplay.features import TargetEncoder, mol_to_graph, morgan_fingerprint
from molplay.utils import scaffold_split


def test_mol_to_graph_shapes():
    data = mol_to_graph("CCO")
    assert data is not None
    assert data.x.shape[0] == 3
    assert data.edge_index.shape[1] == 4  # undirected edges


def test_morgan_fingerprint_length():
    fp = morgan_fingerprint("CCO", radius=2, n_bits=1024)
    assert fp is not None
    assert fp.shape == (1024,)
    assert fp.dtype == np.float32


def test_target_encoder_one_hot():
    encoder = TargetEncoder(method="one_hot")
    encoder.fit(["A", "B", "A"], [900, 1100, 900])
    transformed = encoder.transform(["A", "B"])
    assert transformed.shape == (2, 2)
    assert transformed[0, 0] == 1.0
    assert transformed[1, 1] == 1.0


def test_scaffold_split_partitions():
    smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccccc1O"]
    split = scaffold_split(smiles, frac_train=0.5, frac_valid=0.25, frac_test=0.25, seed=1)
    train_set = set(split.train)
    valid_set = set(split.valid)
    test_set = set(split.test)
    assert train_set.isdisjoint(valid_set)
    assert train_set.isdisjoint(test_set)
    assert valid_set.isdisjoint(test_set)
    assert len(train_set | valid_set | test_set) == len(smiles)

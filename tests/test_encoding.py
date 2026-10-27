import numpy as np
from dedup_lab.encoding import hashing_embeddings, normalize_rows
def test_embeddings_normalized_and_deterministic():
    texts=["blue cotton shirt","blue cotton shirt","red running shoes"]
    a=hashing_embeddings(texts,128); b=hashing_embeddings(texts,128)
    assert a.shape==(3,128) and np.allclose(a,b)
    assert np.allclose(np.linalg.norm(a,axis=1),1)
def test_empty_text_is_finite():
    x=hashing_embeddings([""],32)
    assert np.isfinite(x).all() and np.linalg.norm(x)==0

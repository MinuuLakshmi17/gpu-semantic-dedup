import numpy as np
from dedup_lab.backends import similarity_topk_numpy
def test_topk_finds_exact_duplicate_and_excludes_self():
    x=np.array([[1,0],[1,0],[0,1]],dtype=np.float32)
    pairs=similarity_topk_numpy(x,top_k=1,chunk_size=2,threshold=.99)
    assert any(i==0 and j==1 and score>.99 for i,j,score in pairs)
    assert all(i!=j for i,j,_ in pairs)
def test_chunk_sizes_agree_on_simple_case():
    x=np.array([[1,0],[.99,.01],[0,1],[-1,0]],dtype=np.float32)
    a=similarity_topk_numpy(x,top_k=2,chunk_size=1,threshold=.8)
    b=similarity_topk_numpy(x,top_k=2,chunk_size=3,threshold=.8)
    assert {(i,j) for i,j,_ in a}=={(i,j) for i,j,_ in b}
def test_single_row_has_no_candidates():
    assert similarity_topk_numpy(np.array([[1,0]],dtype=np.float32))==[]

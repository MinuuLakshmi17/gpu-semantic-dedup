from __future__ import annotations
import numpy as np
from .encoding import normalize_rows

def similarity_topk_numpy(embeddings: np.ndarray, top_k: int=5, chunk_size: int=512,
                          threshold: float=0.88) -> list[tuple[int,int,float]]:
    x=normalize_rows(embeddings)
    n=x.shape[0]
    if top_k < 1 or chunk_size < 1: raise ValueError("top_k and chunk_size must be positive")
    results=[]
    for start in range(0,n,chunk_size):
        end=min(start+chunk_size,n)
        scores=x[start:end] @ x.T
        for local,row in enumerate(scores):
            i=start+local
            row[i]=-np.inf
            k=min(top_k,max(0,n-1))
            if k==0: continue
            idx=np.argpartition(row,-k)[-k:]
            idx=idx[np.argsort(row[idx])[::-1]]
            results.extend((i,int(j),float(row[j])) for j in idx if row[j]>=threshold)
    return results

def similarity_topk_torch(embeddings: np.ndarray, top_k=5, chunk_size=512, threshold=.88,
                          device="cuda", fp16=False):
    try: import torch
    except ImportError as exc: raise RuntimeError("Install PyTorch to use backend=torch") from exc
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    x=torch.as_tensor(embeddings,device=device,dtype=torch.float32)
    x=torch.nn.functional.normalize(x,dim=1)
    n=x.shape[0]; out=[]
    for start in range(0,n,chunk_size):
        part=x[start:min(start+chunk_size,n)]
        if fp16: scores=(part.half() @ x.half().T).float()
        else: scores=part @ x.T
        rows=torch.arange(len(part),device=device); cols=torch.arange(start,start+len(part),device=device)
        scores[rows,cols]=-torch.inf
        k=min(top_k,max(0,n-1))
        if not k: continue
        vals,inds=torch.topk(scores,k=k,dim=1)
        # Transfer each bounded chunk only once.
        vals_cpu=vals.detach().float().cpu().numpy(); inds_cpu=inds.detach().cpu().numpy()
        for local in range(len(part)):
            i=start+local
            out.extend((i,int(j),float(v)) for j,v in zip(inds_cpu[local],vals_cpu[local]) if v>=threshold)
    return out

def similarity_topk_cupy(embeddings: np.ndarray, top_k=5, chunk_size=512, threshold=.88, device="cuda"):
    try: import cupy as cp
    except ImportError as exc: raise RuntimeError("Install a CUDA-matching CuPy package to use backend=cupy") from exc
    if not device.startswith("cuda"): raise ValueError("CuPy backend requires device=cuda")
    x=cp.asarray(normalize_rows(embeddings))
    n=x.shape[0]; out=[]
    for start in range(0,n,chunk_size):
        end=min(start+chunk_size,n); scores=x[start:end] @ x.T
        rows=cp.arange(end-start); cols=cp.arange(start,end); scores[rows,cols]=-cp.inf
        k=min(top_k,max(0,n-1))
        if not k: continue
        idx=cp.argpartition(scores,-k,axis=1)[:,-k:]
        vals=cp.take_along_axis(scores,idx,axis=1)
        order=cp.argsort(vals,axis=1)[:,::-1]
        idx=cp.take_along_axis(idx,order,axis=1); vals=cp.take_along_axis(vals,order,axis=1)
        idx_cpu=cp.asnumpy(idx); vals_cpu=cp.asnumpy(vals)
        for local in range(end-start):
            out.extend((start+local,int(j),float(v)) for j,v in zip(idx_cpu[local],vals_cpu[local]) if v>=threshold)
    return out

def normalize_triton(embeddings: np.ndarray) -> np.ndarray:
    try:
        import torch, triton, triton.language as tl
    except ImportError as exc: raise RuntimeError("Install PyTorch and Triton on a compatible Linux/CUDA setup") from exc
    if not torch.cuda.is_available(): raise RuntimeError("Triton normalization requires CUDA")
    @triton.jit
    def kernel(X,Y,N:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
        row=tl.program_id(0); cols=tl.arange(0,B); vals=tl.load(X+row*D+cols,cols<D,0)
        norm=tl.sqrt(tl.sum(vals*vals,axis=0)); out=vals/tl.maximum(norm,1.e-12)
        tl.store(Y+row*D+cols,out,cols<D)
    arr=torch.as_tensor(embeddings,device="cuda",dtype=torch.float32).contiguous()
    n,d=arr.shape
    if d > 4096: raise ValueError("Triton kernel supports dimensions <= 4096")
    out=torch.empty_like(arr); kernel[(n,)](arr,out,n,d,triton.next_power_of_2(d))
    return out.cpu().numpy()

def similarity_topk_triton(embeddings, top_k=5, chunk_size=512, threshold=.88, device="cuda"):
    # Triton kernel optimizes row normalization; chunked matmul/top-k uses PyTorch.
    normalized=normalize_triton(embeddings)
    return similarity_topk_torch(normalized,top_k,chunk_size,threshold,device=device)

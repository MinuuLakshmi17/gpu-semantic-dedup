from __future__ import annotations
import hashlib, re
import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")

def hashing_embeddings(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic signed feature hashing baseline; output rows are L2 normalized."""
    if dim < 8: raise ValueError("dim must be >= 8")
    out=np.zeros((len(texts),dim),dtype=np.float32)
    for i,text in enumerate(texts):
        tokens=TOKEN_RE.findall((text or "").lower())
        # Include adjacent token pairs to capture simple phrase similarity.
        features=tokens + [tokens[j]+"_"+tokens[j+1] for j in range(len(tokens)-1)]
        for token in features:
            digest=hashlib.blake2b(token.encode("utf-8"),digest_size=8).digest()
            value=int.from_bytes(digest[:4],"little")%dim
            sign=1.0 if digest[4]&1 else -1.0
            out[i,value]+=sign
    return normalize_rows(out)

def normalize_rows(array: np.ndarray) -> np.ndarray:
    array=np.asarray(array,dtype=np.float32)
    norms=np.linalg.norm(array,axis=1,keepdims=True)
    return array/np.maximum(norms,1e-12)

def sentence_transformer_embeddings(texts: list[str], model_name: str, batch_size: int=64) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install -e '.[nlp]'") from exc
    model=SentenceTransformer(model_name)
    result=model.encode(texts,batch_size=batch_size,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
    return np.asarray(result,dtype=np.float32)

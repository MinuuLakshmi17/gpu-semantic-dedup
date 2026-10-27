from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
from .encoding import hashing_embeddings, sentence_transformer_embeddings
from .backends import similarity_topk_numpy, similarity_topk_torch, similarity_topk_cupy, similarity_topk_triton

def load_products(path: str):
    path=Path(path); rows=[]
    if path.suffix.lower()==".csv":
        with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    elif path.suffix.lower() in {".jsonl",".ndjson"}:
        with path.open(encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                if line.strip():
                    try: rows.append(json.loads(line))
                    except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSONL at line {line_no}") from exc
    elif path.suffix.lower()==".json":
        data=json.loads(path.read_text(encoding="utf-8"))
        rows=data if isinstance(data,list) else data.get("products",[])
    else: raise ValueError("Input must be .jsonl, .ndjson, .json or .csv")
    cleaned=[]
    for i,row in enumerate(rows):
        title=str(row.get("title",row.get("name",""))).strip()
        if not title: continue
        cleaned.append({"id":str(row.get("id",i+1)),"title":title})
    if not cleaned: raise ValueError("No rows with a non-empty title found")
    return cleaned

def run_pipeline(rows, backend="numpy", top_k=5, threshold=.88, chunk_size=512,
                 dim=384, encoder="hashing", model_name=None, device="cuda", fp16=False):
    texts=[r["title"] for r in rows]
    start=time.perf_counter()
    if encoder=="hashing": embeddings=hashing_embeddings(texts,dim)
    elif encoder=="sentence-transformers":
        if not model_name: raise ValueError("--model is required for sentence-transformers encoder")
        embeddings=sentence_transformer_embeddings(texts,model_name)
    else: raise ValueError(f"Unknown encoder: {encoder}")
    encode_seconds=time.perf_counter()-start
    start=time.perf_counter()
    fn={"numpy":similarity_topk_numpy,"torch":similarity_topk_torch,"cupy":similarity_topk_cupy,"triton":similarity_topk_triton}.get(backend)
    if fn is None: raise ValueError(f"Unknown backend: {backend}")
    kwargs={"top_k":top_k,"chunk_size":chunk_size,"threshold":threshold}
    if backend=="torch": kwargs.update(device=device,fp16=fp16)
    elif backend in {"cupy","triton"}: kwargs.update(device=device)
    pairs=fn(embeddings,**kwargs)
    sim_seconds=time.perf_counter()-start
    result=[]
    for i,j,score in pairs:
        result.append({"source_id":rows[i]["id"],"candidate_id":rows[j]["id"],"source_title":rows[i]["title"],
                       "candidate_title":rows[j]["title"],"cosine_similarity":round(score,6)})
    return result,{"rows":len(rows),"embedding_dim":int(embeddings.shape[1]),"encoder":encoder,"backend":backend,
                   "encode_seconds":round(encode_seconds,4),"similarity_seconds":round(sim_seconds,4),
                   "pairs":len(result),"chunk_size":chunk_size,"top_k":top_k,"threshold":threshold}

def write_results(results, output):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    if output.suffix.lower()==".csv":
        fields=["source_id","candidate_id","source_title","candidate_title","cosine_similarity"]
        with output.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
    elif output.suffix.lower()==".parquet":
        try: import pandas as pd
        except ImportError as exc: raise RuntimeError("Parquet output requires pandas and pyarrow; use a .csv output or pip install -e '.[dev]'") from exc
        pd.DataFrame(results,columns=["source_id","candidate_id","source_title","candidate_title","cosine_similarity"]).to_parquet(output,index=False)
    else: raise ValueError("Output extension must be .csv or .parquet")

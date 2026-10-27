from __future__ import annotations
import argparse, json, platform, importlib.util
from pathlib import Path
import numpy as np
from .encoding import hashing_embeddings
from .pipeline import load_products, run_pipeline, write_results
from .backends import similarity_topk_numpy

def generate(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    brands=["Northstar","BluePeak","UrbanField","Everwell","NovaHome","TrailCraft"]
    items=["cotton crew neck t shirt","wireless bluetooth headphones","stainless steel water bottle","ergonomic office chair","ceramic coffee mug","running shoes"]
    with path.open("w",encoding="utf-8") as f:
        for i in range(rows):
            base=items[i%len(items)]; brand=brands[(i//3)%len(brands)]
            title=f"{brand} {base} {['black','blue','large','new edition','premium'][i%5]}"
            if i%17==0 and i>0: title=f"{brand} {items[(i-1)%len(items)]} {['black','blue','large','new edition','premium'][(i-1)%5]}"
            f.write(json.dumps({"id":f"SKU-{i+1:08d}","title":title})+"\n")
    print(f"Wrote {rows} synthetic listings to {path}")

def main(argv=None):
    parser=argparse.ArgumentParser(prog="dedup-lab",description="Chunked product semantic deduplication lab")
    sub=parser.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate");g.add_argument("--rows",type=int,default=10000);g.add_argument("--out",default="data/products.jsonl")
    r=sub.add_parser("run");r.add_argument("--input",required=True);r.add_argument("--output",required=True)
    r.add_argument("--backend",choices=["numpy","torch","cupy","triton"],default="numpy");r.add_argument("--device",default="cuda")
    r.add_argument("--top-k",type=int,default=5);r.add_argument("--threshold",type=float,default=.88);r.add_argument("--chunk-size",type=int,default=512)
    r.add_argument("--dim",type=int,default=384);r.add_argument("--encoder",choices=["hashing","sentence-transformers"],default="hashing")
    r.add_argument("--model");r.add_argument("--fp16",action="store_true");r.add_argument("--undirected",action="store_true")
    b=sub.add_parser("benchmark");b.add_argument("--rows",type=int,default=3000);b.add_argument("--backend",choices=["numpy","torch"],default="numpy");b.add_argument("--chunk-size",type=int,default=256)
    sub.add_parser("doctor")
    args=parser.parse_args(argv)
    if args.command=="generate":
        if args.rows<1: parser.error("--rows must be positive")
        generate(args.out,args.rows);return
    if args.command=="doctor":
        print(json.dumps({"python":platform.python_version(),"numpy":np.__version__,"torch":bool(importlib.util.find_spec("torch")),
                          "cupy":bool(importlib.util.find_spec("cupy")),"triton":bool(importlib.util.find_spec("triton")),
                          "sentence_transformers":bool(importlib.util.find_spec("sentence_transformers"))},indent=2));return
    if args.command=="benchmark":
        from time import perf_counter
        rows=[{"id":str(i),"title":f"demo item category {i%100} color {i%7}"} for i in range(args.rows)]
        _,metrics=run_pipeline(rows,backend=args.backend,chunk_size=args.chunk_size,top_k=5,threshold=.9,device="cuda")
        print(json.dumps(metrics,indent=2));return
    if args.top_k<1 or args.chunk_size<1 or not -1<=args.threshold<=1: parser.error("top-k/chunk-size must be positive and threshold must be between -1 and 1")
    rows=load_products(args.input)
    results,metrics=run_pipeline(rows,backend=args.backend,top_k=args.top_k,threshold=args.threshold,chunk_size=args.chunk_size,
                                 dim=args.dim,encoder=args.encoder,model_name=args.model,device=args.device,fp16=args.fp16)
    if args.undirected:
        dedup={}
        for p in results:
            a,b=sorted([p["source_id"],p["candidate_id"]]); key=(a,b)
            if key not in dedup or p["cosine_similarity"]>dedup[key]["cosine_similarity"]:
                dedup[key]={**p,"source_id":a,"candidate_id":b}
        results=list(dedup.values())
    write_results(results,args.output)
    metrics["output"]=args.output;metrics["undirected"]=args.undirected
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()

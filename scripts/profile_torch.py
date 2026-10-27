import argparse, json, time
import numpy as np
try: import torch
except ImportError as exc: raise SystemExit("Install PyTorch first") from exc
p=argparse.ArgumentParser();p.add_argument("--rows",type=int,default=10000);p.add_argument("--dim",type=int,default=384);a=p.parse_args()
device="cuda" if torch.cuda.is_available() else "cpu"
x=torch.randn(a.rows,a.dim,device=device); y=torch.nn.functional.normalize(x,dim=1)
if device=="cuda": torch.cuda.synchronize()
start=time.perf_counter()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if device=="cuda" else []),record_shapes=True,profile_memory=True) as prof:
    for i in range(0,a.rows,512):
        _=y[i:i+512] @ y.T
        if device=="cuda": torch.cuda.synchronize()
print(json.dumps({"device":device,"rows":a.rows,"seconds":time.perf_counter()-start},indent=2))
print(prof.key_averages().table(sort_by="self_cuda_time_total" if device=="cuda" else "self_cpu_time_total",row_limit=20))
prof.export_chrome_trace("torch-profiler-trace.json")
print("Wrote torch-profiler-trace.json")

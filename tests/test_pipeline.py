import json
from dedup_lab.pipeline import load_products, run_pipeline, write_results
def test_load_jsonl_and_run(tmp_path):
    p=tmp_path/"products.jsonl"
    p.write_text(json.dumps({"id":"a","title":"blue shirt"})+"\n"+json.dumps({"id":"b","title":"blue shirt"})+"\n")
    rows=load_products(str(p))
    result,metrics=run_pipeline(rows,top_k=1,threshold=.99,dim=64,chunk_size=1)
    assert metrics["rows"]==2
    assert any(r["source_id"]=="a" and r["candidate_id"]=="b" for r in result)
    out=tmp_path/"out.csv";write_results(result,str(out));assert out.exists()
def test_bad_jsonl_has_line_number(tmp_path):
    p=tmp_path/"bad.jsonl";p.write_text('not-json\n')
    try: load_products(str(p))
    except ValueError as e: assert "line 1" in str(e)
    else: assert False

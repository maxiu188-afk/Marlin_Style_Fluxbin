#!/usr/bin/env python3
"""Summarize saved candidate JSON/logs offline, including failed/missing cells."""
import argparse
import json
from pathlib import Path
from fluxbin_style.candidate_summary import summarize_batch,markdown

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/acceleration/m1_candidates_v1.json')
    args=p.parse_args()
    report=summarize_batch(args.batch_dir,args.config)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output_dir/'summary.md').write_text(markdown(report))
    print(f"{sum(r['eligible'] for r in report['rows'])}/{len(report['rows'])} eligible cells; review required")

if __name__=='__main__':main()

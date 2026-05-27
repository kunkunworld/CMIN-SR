from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.data import RawZetaAlignmentDatasetConfig, generate_raw_zeta_alignment_dataset
def main():
    ds=generate_raw_zeta_alignment_dataset(RawZetaAlignmentDatasetConfig(length=256,num_samples=64,seed=1))
    result={"status":"ok","x_shape":list(ds["x"].shape),"processes":sorted(set(ds["process_type"].tolist())),"target_shape":list(ds["zeta_target"].shape)}
    p=ROOT/"outputs"/"reports"/"raw_zeta_alignment_dataset_smoke_test.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()

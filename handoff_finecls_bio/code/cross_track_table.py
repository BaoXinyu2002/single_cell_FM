"""projector-only vs refiner, on IDENTICAL cells, identical pools, identical centring.

⛔ THIS SCORER APPLIES DONOR CENTRING BUT **NOT** CSLS.  Both are inference-time levers and
   together they are worth ~2x on OOD R@1 (bare 0.1040 -> +CSLS10 0.1314 -> +donor centring
   0.1799 -> both 0.2065, 4 true-OOD sets).  Every number this file prints is therefore on
   the "donor, no CSLS" rung and understates the arms by roughly +0.027.  Do not compare its
   output against numbers scored with CSLS -- see README.md sec. 3b.

⛔ BARCODE INTERSECTION IS MANDATORY ON breast AND liver.  The refiner's cells are a strict
   SUBSET of the projector's there (8859/9739, 29922/30135).  Measured: not intersecting
   moves breast a2r@1 by +0.0039, LARGER than the registered MDE of 0.0023, and it reads as
   a projector advantage that is purely a different cell set.  The other four sets match
   exactly and are passed through unchanged (asserted, not assumed).
⛔ DIRECTIONS ARE NEVER AVERAGED.
⛔ islet and pln are TRAINING donors -- reported apart, never inside an OOD mean.
⛔ Pools are dataset-window, 128 cells, 200 draws, crc32-seeded PER DATASET so every arm on
   both tracks sees the SAME pools.  The pools are drawn AFTER the intersection, so the two
   tracks index the same rows.
"""
import numpy as np, pandas as pd, os, sys, json, zlib
B="/nfs/turbo/umms-drjieliu1/usr/xinyubao/sclip"
H=f"{B}/experiments/finecls_refiner"
POOL,DRAWS = 128,200
OOD4=["bmmc","breast","fetal_heart","liver"]; ID2=["islet","pln"]
NEED_INTERSECT={"breast","liver"}
l2=lambda x: x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

REFINER={  # arm -> emb dir
 "A_cell_only":       f"{H}/runs/seed0_conv/cell_only/emb_cooled",
 "B_finecls_bio":     f"{H}/runs/seed0_conv/finecls_bio/emb_cooled",
 "C_random64":        f"{H}/runs/seed0_conv/finecls_random64/emb_cooled",
 "D_bio_fused":       f"{H}/runs/fused/finecls_bio_fused/emb_cooled",
 "E_random64_fused":  f"{H}/runs/fused/finecls_random64_fused/emb_cooled",
}
PROJECTOR={  # train-time centring NOT adopted -> c0 is the representative; c1 kept for the record
 "P_c0_nocenter": f"{B}/experiments/results/valfix_ood/c0_step034000",
 "P_c1_center":   f"{B}/experiments/results/valfix_ood/c1_step034000",
}

def load_refiner(d,ds):
    p=f"{d}/{ds}.npz"
    if not os.path.isfile(p): return None
    z=np.load(p,allow_pickle=True)
    return dict(bc=np.asarray(z["barcode"]).astype(str),
                r=np.asarray(z["rna"],np.float32), a=np.asarray(z["atac"],np.float32),
                donor=np.asarray(z["donor"]).astype(str))

def load_projector(d,ds):
    b=f"{d}/{ds}"
    if not os.path.isfile(f"{b}/clip_rna_embeddings.npy"): return None
    i=pd.read_csv(f"{b}/rna_cell_info.csv")
    dn=next((c for c in ("donor","batch","dataset_id") if c in i.columns),None)
    return dict(bc=i["cell_barcode"].astype(str).values,
                r=np.load(f"{b}/clip_rna_embeddings.npy").astype(np.float32),
                a=np.load(f"{b}/clip_atac_embeddings.npy").astype(np.float32),
                donor=i[dn].astype(str).values if dn else None)

def center(x,g):
    if g is None: return x
    out=x.copy()
    for u in np.unique(g):
        m=g==u
        if m.sum()>1: out[m]=x[m]-x[m].mean(0,keepdims=True)
    return l2(out)

def recalls(r,a,pools,ks=(1,10)):
    acc={f"{d}@{k}":[] for k in ks for d in ("r2a","a2r")}
    for p in pools:
        S=r[p]@a[p].T; P=len(p); d=np.arange(P)
        for k in ks:
            kk=min(k,P)
            acc[f"r2a@{k}"].append(float((np.argpartition(-S,kk-1,axis=1)[:,:kk]==d[:,None]).any(1).mean()))
            acc[f"a2r@{k}"].append(float((np.argpartition(-S,kk-1,axis=0)[:kk,:].T==d[:,None]).any(1).mean()))
    return {k:float(np.mean(v)) for k,v in acc.items()}

rows=[]; missing=[]
for ds in OOD4+ID2:
    arms={}
    for lab,d in REFINER.items():
        o=load_refiner(d,ds)
        if o is None: missing.append(f"{lab}/{ds}"); continue
        arms[("refiner",lab)]=o
    for lab,d in PROJECTOR.items():
        o=load_projector(d,ds)
        if o is None: missing.append(f"{lab}/{ds}"); continue
        arms[("projector",lab)]=o
    if not arms: continue
    # --- the shared cell set ---
    common=None
    for o in arms.values():
        s=pd.Index(o["bc"]); common = s if common is None else common.intersection(s)
    if ds not in NEED_INTERSECT:
        for k,o in arms.items():
            assert len(o["bc"])==len(common), f"{ds} {k}: expected identical cell sets, got {len(o['bc'])} vs {len(common)}"
    common=common.sort_values()
    n=len(common)
    rs=np.random.RandomState(zlib.crc32(ds.encode())&0x7fffffff)
    pools=[rs.choice(n,min(POOL,n),replace=False) for _ in range(DRAWS)]
    for (track,lab),o in arms.items():
        pos=pd.Index(o["bc"]).get_indexer(common)
        assert (pos>=0).all(), f"{ds} {lab}: barcode lookup failed"
        r,a=l2(o["r"][pos]),l2(o["a"][pos])
        dn=o["donor"][pos] if o["donor"] is not None else None
        for mode,g in [("off",None),("dataset",np.zeros(n,int)),("donor",dn)]:
            if mode=="donor" and dn is None: continue
            m=recalls(center(r,g),center(a,g),pools)
            rows.append(dict(dataset=ds,split=("OOD" if ds in OOD4 else "ID_train_donor"),
                             track=track,arm=lab,center=mode,n=n,**m))
df=pd.DataFrame(rows)
out=f"{H}/results/cross_track"; os.makedirs(out,exist_ok=True)
df.to_csv(f"{out}/table.csv",index=False)
print(f"wrote {len(df)} rows -> {out}/table.csv")
if missing: print("⛔ MISSING (arm/dataset):", ", ".join(sorted(set(missing))))

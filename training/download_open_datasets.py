import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='mohanty/PlantVillage'); p.add_argument('--config',default='color'); p.add_argument('--out',default='data/raw/plantvillage'); a=p.parse_args()
    try: from datasets import load_dataset
    except ImportError: raise SystemExit('Install datasets first: pip install datasets huggingface_hub')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); ds=load_dataset(a.dataset,a.config)
    for split,rows in ds.items():
        names=getattr(ds[split].features.get('label'),'names',None)
        for i,row in enumerate(rows):
            lab=row.get('label'); lab=names[lab] if isinstance(lab,int) and names else lab; img=row.get('image')
            if lab and img is not None: d=out/str(lab); d.mkdir(parents=True,exist_ok=True); img.save(d/f'{split}_{i}.jpg')
    print('Exported PlantVillage to',out.resolve())
if __name__=='__main__': main()

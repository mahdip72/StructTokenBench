# Simplified Scripts (lite embeddings)

These commands assume:
- Python env: `source /mnt/hdd8/mehdi/environments/StructTokenBench/bin/activate`
- Python: `python`
- H5 embeddings: `/mnt/hdd8/farzaneh/projects/PST/embeddings/`
- Data roots:
  - Functional: `/mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local`
  - Structural: `/mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/structural`
  - Physicochemical (ATLAS): `/mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/physicochemical`
  - Sensitivity (Apo/Holo): `/mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/sensitivity`

Adjust paths if your local layout is different.

## BioLIP2 binding (lite)
```bash
python new_scripts/biolip2_binding_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_biolip2_binding_lite_model.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local
```

## BioLIP2 catalytic (lite)
```bash
python new_scripts/biolip2_catalytic_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_biolip2_catalytic_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local
```

## ProteinShake binding site (lite)
```bash
python new_scripts/proteinshake_binding_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_proteinshake_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local
```

## InterPro binding (lite)
```bash
python new_scripts/interpro_binding_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_interpro_binding_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local
```

## InterPro active site (lite)
```bash
python new_scripts/interpro_activesite_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_interpro_activesite_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local
```

## InterPro conserved site (lite)
```bash
python new_scripts/interpro_conserved_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_conserved_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local
```

## ProteinGLUE epitope region (lite)
```bash
python new_scripts/proteinglue_epitope_region_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_proteinglue_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local
```

## ATLAS RMSF (lite)
```bash
python new_scripts/atlas_rmsf_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_atlas_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/physicochemical
```

## ATLAS B-factor (lite)
```bash
python new_scripts/atlas_bfactor_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_atlas_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/physicochemical
```

## ATLAS NEQ (lite)
```bash
python new_scripts/atlas_neq_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_atlas_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/physicochemical
```

## Remote homology (lite)
```bash
python new_scripts/remote_homology_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_remote_homology_train_tst_valid.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/structural \
  --progress
```

## Apo/Holo + Fold Switching (unsupervised, lite)
```bash
python new_scripts/apolo_unsupervised_eval.py \
  --h5 /mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_apolo_lite.h5 \
  --embeddings-dataset / \
  --data-root /mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/sensitivity \
  --target-field tm_score
```

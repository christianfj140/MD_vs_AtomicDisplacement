# graphene_w90_joint

Default local dataset root for the one-click Graph2Mat vs DeepH graphene
workflow.

Place or generate the validated joint SIESTA/Wannier90 snapshots here. A ready
dataset should contain, at minimum:

- `MD_steps/<sample>/RUN.fdf`
- `MD_steps/<sample>/RUN.out` or `siesta.out`
- `MD_steps/<sample>/graphene.TSHS`
- `MD_steps/<sample>/graphene.TSDE`
- `MD_steps/<sample>/graphene.HSX`
- `MD_steps/<sample>/graphene.STRUCT_OUT`
- `MD_steps/<sample>/graphene.XV`
- `MD_steps/<sample>/graphene.ORB_INDX`
- `MD_steps/<sample>/metadata.json`
- `artifact_validation.json`
- `benchmark_dataset_manifest.json`
- `frozen_split_manifest.json`

The normal benchmark workflow must validate this directory and must not repair
missing DeepH artifacts with silent per-snapshot SIESTA reruns.

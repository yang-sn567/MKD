# Dataset Notes

This packaged project includes one compact local CSV file:

```text
dataset/data_100k_with_llm_text.csv
```

- `dataset/data_100k_with_llm_text.csv`: compact 100K PCQM4Mv2-style subset with molecule index, SMILES, target value, and an additional `llm_text` column for the LLM Text modality.

The full original PCQM4Mv2 1D dataset is not packaged. The project keeps dataset paths configurable, and large official resources are not generated automatically.

For PCQM4Mv2 official resources, refer to:

https://ogb.stanford.edu/docs/lsc/pcqm4mv2/

The OGB documentation describes:

- Original 1D SMILES/target data.
- 2D molecular graph construction from SMILES through `smiles2graph`.
- Graph dictionary fields such as `edge_index`, `edge_feat`, `node_feat`, and `num_nodes`.
- Official train/validation/test-dev/test-challenge split.
- 3D conformer data distributed as the `pcqm4m-v2-train.sdf` file.

# MKD: Multimodal Knowledge Distillation For Molecular Property Prediction

![MKD framework](assets/framework.png)

For 2D/3D training, make sure these packages are available:

- `torch`
- `torch-geometric`
- `ogb`
- `rdkit`
- `numpy`
- `pandas`
- `pyyaml`
- `tqdm`
- `transformers`
- `tokenizers`

## 1. Dataset Files

This project includes the LLM_Text dataset from the paper: ```dataset/data_100k_with_llm_text.csv```. The complete original 1D data of PCQM4Mv2 should be obtained from the official OGB documentation (https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) and the dataset path should be provided through a configuration file or command-line parameters.

## 2. Configs

Main configs:

```text
configs/smoke.yaml
configs/pcqm4mv2_1d2d_text.yaml
configs/pcqm4mv2_1d2d3d_text.yaml
configs/pcqm4mv2_100k.yaml
configs/pcqm4mv2_full.yaml
```

Important fields:

```yaml
data:
  one_d_path: dataset/data_100k.csv
  graph_path: pcqm4mv2
  geometry_path:
  llm_text_path: dataset/llm_text.csv
  smiles_column: smiles
  label_column: homolumogap
  text_column: llm_text
  task_type: regression
  split_method: official_pcqm4mv2
  split_root: dataset
```

If a modality path is omitted, that modality is not trained and is not sent to its teacher or student branch.

The default LLM Text encoder uses ChemBERTa through HuggingFace:

```yaml
model:
  text:
    encoder_type: chemberta
    pretrained_model_name: DeepChem/ChemBERTa-77M-MTR
    freeze_encoder: true
    use_safetensors: true
    pooling: cls
    max_length: 256
```

The first run downloads the model weights into the local HuggingFace cache. After that, the cached model can be reused.

To create a compact PCQM4Mv2 subset while preserving the official 90/2/4/4 split ratio:

```bash
python scripts/create_pcqm4mv2_subset.py ^
  --input-csv dataset/data.csv ^
  --output-csv dataset/data_100k.csv ^
  --split-root dataset ^
  --sample-size 100000
```

## 4. Official Split

The project follows the official PCQM4Mv2 split method:

```python
split_dict = dataset.get_idx_split()
train_idx = split_dict["train"]
valid_idx = split_dict["valid"]
testdev_idx = split_dict["test-dev"]
testchallenge_idx = split_dict["test-challenge"]
```

For training and early stopping, the project uses:

- `train`
- `valid`

For this repository, validation is used to select checkpoints by training loss. The core pipeline focuses on demonstrating the multimodal teacher-student framework instead of maintaining a benchmark suite.

## 5. Run Training

Run the tiny example pipeline:

```bash
python scripts/run_experiment.py --config configs/smoke.yaml
```

Run the PCQM4Mv2 teacher + student pipeline:

```bash
python scripts/run_experiment.py --config configs/pcqm4mv2_1d2d_text.yaml
```

Override modality paths from the command line:

```bash
python scripts/run_experiment.py ^
  --config configs/pcqm4mv2_1d2d_text.yaml ^
  --one-d-path dataset/data_100k.csv ^
  --graph-path pcqm4mv2 ^
  --llm-text-path dataset/llm_text.csv
```

With 3D SDF:

```bash
python scripts/run_experiment.py ^
  --config configs/pcqm4mv2_1d2d3d_text.yaml ^
  --one-d-path dataset/data.csv ^
  --graph-path pcqm4mv2 ^
  --geometry-path path/to/pcqm4m-v2-train.sdf ^
  --llm-text-path path/to/llm_text.csv
```

## 6. Checkpoints And Outputs

Teacher checkpoints:

```text
checkpoints/.../smiles_teacher.pt
checkpoints/.../graph_teacher.pt
checkpoints/.../geometry_teacher.pt
checkpoints/.../llm_text_teacher.pt
checkpoints/.../mkd_student.pt
```

Last-epoch checkpoints are also written for recovery or inspection:

```text
checkpoints/.../*_teacher_last.pt
checkpoints/.../mkd_student_last.pt
```

Outputs:

```text
outputs/.../resolved_config.json
outputs/.../results.json
outputs/.../train_log.jsonl
outputs/.../student_valid_loss.json
```

Student training requires teacher checkpoints for every active modality when:

```yaml
require_teacher_checkpoints: true
```

## 7. Separate Teacher Or Student Training

Train teachers only:

```bash
python scripts/train_teachers.py --config configs/pcqm4mv2_1d2d_text.yaml
```

Train student only after teacher checkpoints exist:

```bash
python scripts/train_mkd.py --config configs/pcqm4mv2_1d2d_text.yaml
```

## 8. What To Provide For A New Dataset

At minimum:

- A CSV with `idx`, `smiles`, and target label.

Optional:

- `--graph-path pcqm4mv2` for 2D graph training.
- A full SDF file for 3D SchNet training.
- A separate `idx,llm_text` CSV for semantic text training.

The project trains only the modalities whose inputs are provided.

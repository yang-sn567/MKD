from mkd_mpp.config import load_config
from mkd_mpp.models import HuggingFaceTextEncoder, MKDStudent, SmilesTransformerEncoder, build_text_encoder
from mkd_mpp.pcqm4mv2 import smiles2graph


def test_config_loads():
    config = load_config("configs/default.yaml")
    assert config.model.hidden_dim > 0


def test_model_constructs():
    model = MKDStudent(modalities=["smiles"], hidden_dim=32, output_dim=1)
    assert model is not None


def test_smiles_encoder_constructs():
    encoder = SmilesTransformerEncoder(hidden_dim=32, num_layers=1, num_heads=4)
    assert encoder is not None


def test_llm_text_encoder_class_imports():
    assert HuggingFaceTextEncoder is not None


def test_text_encoder_factory_rejects_unsupported_encoder():
    try:
        build_text_encoder(32, config={"encoder_type": "transformer"})
    except ValueError:
        return
    raise AssertionError("unsupported text encoder should raise ValueError")


def test_pcqm4mv2_smiles2graph_imports():
    assert callable(smiles2graph)

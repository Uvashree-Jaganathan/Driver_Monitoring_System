import importlib.util
from pathlib import Path


def test_training_script_exists_and_exposes_main():
    script_path = Path("scripts/train_eye_cnn.py")
    spec = importlib.util.spec_from_file_location("train_eye_cnn", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "main")
    assert callable(module.main)

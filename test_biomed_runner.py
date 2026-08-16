import importlib.util
from pathlib import Path


def test_biomed_runner_module_exports():
    path = Path(__file__).with_name("biomed_runner.py")
    spec = importlib.util.spec_from_file_location("biomed_runner", path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "load_biomedclip_model")
    assert hasattr(module, "encode_biomedclip_image")
    assert hasattr(module, "encode_biomedclip_images")

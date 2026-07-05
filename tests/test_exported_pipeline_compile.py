import py_compile
from pathlib import Path


def test_exported_training_pipeline_compiles():
    py_compile.compile(
        str(Path("src/training_pipeline_exported.py")),
        doraise=True,
    )

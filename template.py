import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "image_captioning_system"

list_of_files = [

    # GitHub
    ".github/workflows/.gitkeep",

    # Core package
    f"src/{project_name}/__init__.py",

    # Data pipeline
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/data_ingestion.py",
    f"src/{project_name}/data/data_preprocessing.py",
    f"src/{project_name}/data/feature_extraction.py",

    # Model
    f"src/{project_name}/model/__init__.py",
    f"src/{project_name}/model/model.py",
    f"src/{project_name}/model/train.py",
    f"src/{project_name}/model/evaluate.py",

    # Inference
    f"src/{project_name}/inference/__init__.py",
    f"src/{project_name}/inference/greedy.py",
    f"src/{project_name}/inference/beam_search.py",

    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/metrics.py",

    # Pipeline stages (DVC)
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_ingest.py",
    f"src/{project_name}/pipeline/stage_02_preprocess.py",
    f"src/{project_name}/pipeline/stage_03_features.py",
    f"src/{project_name}/pipeline/stage_04_train.py",
    f"src/{project_name}/pipeline/stage_05_evaluate.py",

    # Config
    "config/config.yaml",
    "params.yaml",
    "dvc.yaml",

    # Streamlit
    "streamlit_app/app.py",

    # Meta
    "requirements.txt",
    "setup.py",
    "README.md",

    # Research
    "research/experiments.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if (not filepath.exists()) or (filepath.stat().st_size == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

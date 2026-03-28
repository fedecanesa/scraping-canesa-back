# utils/file_storage.py

import json
import os
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "Salidas de los Agentes")


def ensure_run_dir(run_id: str) -> str:
    run_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_text_output(run_id: str, filename: str, content: str) -> str:
    run_dir = ensure_run_dir(run_id)
    out_path = os.path.join(run_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    return out_path


def save_json_output(run_id: str, filename: str, data: Any) -> str:
    run_dir = ensure_run_dir(run_id)
    out_path = os.path.join(run_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return out_path
from os import path, listdir
from typing import List

from models.images import ImageFormat


def get_models_from_text_file(file_path: str) -> List:
    models = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    for repo_id in lines:
        if repo_id.strip() != "":
            models.append(repo_id.strip())
    return models


def get_image_file_extension(image_format: ImageFormat) -> str:
    if isinstance(image_format, ImageFormat):
        return image_format.value.lower()


def get_files_in_dir(root_dir: str) -> List:
    models = []
    models.append("None")
    for file in listdir(root_dir):
        if file.endswith((".gguf", ".safetensors")):
            models.append(path.join(root_dir, file))
    return models

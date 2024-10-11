# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import os

from huggingface_hub import snapshot_download
from pydantic import BaseModel
import yaml

import folder_paths


BASE_NAME = os.path.realpath(os.path.dirname(__file__))


class ModelDefinition(BaseModel):
    name: str
    repo_id: str
    revision: str|None = None
    use_hf_cache: bool


class ModelManager:
    LIST: list[ModelDefinition] = []
    BY_NAME: dict[str,ModelDefinition] = {}
    CHOICES: list[str] = []

    _MODELS_SUBDIR = 'LLM'
    _ALLOWED_FILES = ['*.json', '*.safetensors']

    def __init__(self, models_subdir: str|None=None, allowed_files: list[str]|None=None):
        if models_subdir is not None:
            self._MODELS_SUBDIR = models_subdir
        if allowed_files is not None:
            self._ALLOWED_FILES = allowed_files

    def _get_models_file(self) -> str:
        models_file = os.path.join(BASE_NAME, 'models.yaml')
        if not os.path.exists(models_file):
            models_file = os.path.join(BASE_NAME, 'models.yaml.default')
        return models_file

    def load(self):
        models_file = self._get_models_file()

        with open(models_file) as inp:
            d = yaml.load(inp, yaml.Loader)
        self._mtime = (models_file, os.path.getmtime(models_file))

        self.LIST = []
        for value in d['models']:
            self.LIST.append(ModelDefinition.model_validate(value))
        if not self.LIST:
            raise RuntimeError('Need at least one model defined')
        self.BY_NAME = { d.name: d for d in self.LIST }
        self.CHOICES = [ d.name for d in self.LIST ]

    def refresh(self):
        models_file = self._get_models_file()
        if self._mtime != (models_file, os.path.getmtime(models_file)):
            self.load()

    def download(self, name: str) -> str:
        model_def = self.BY_NAME[name]

        if os.path.exists(model_def.repo_id):
            # Local path, nothing to do
            return model_def.repo_id

        if model_def.use_hf_cache:
            # Easy peasy
            return snapshot_download(model_def.repo_id, revision=model_def.revision, allow_patterns=self._ALLOWED_FILES)
        else:
            dir_name = '--'.join(model_def.repo_id.split('/'))
            model_path = os.path.join(folder_paths.models_dir, self._MODELS_SUBDIR, dir_name)
            os.makedirs(model_path, exist_ok=True)
            return snapshot_download(model_def.repo_id, revision=model_def.revision, allow_patterns=self._ALLOWED_FILES, local_dir=model_path)

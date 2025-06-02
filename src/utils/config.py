from pathlib import Path
import yaml

class Config:
    def __init__(self, config_file='config/settings.yaml'):
        self.config_file = Path(config_file)
        self.settings = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = value
        self.save_config()

    def save_config(self):
        with open(self.config_file, 'w') as file:
            yaml.dump(self.settings, file)
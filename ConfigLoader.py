import json


class ConfigLoader:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            config = json.load(f)
        self.mode = config['mode']
        self.dt = int(1000 / config['FPS'])
        self.port = f'COM{config["PORT"]}'
        self.baudrate = int(config['BAUDRATE'])
        self.temperature = int(config['TEMPERATURE'])
        self.folder = config['FOLDER']
        self.cosmic_ray_removal = config['COSMICRAYREMOVAL']

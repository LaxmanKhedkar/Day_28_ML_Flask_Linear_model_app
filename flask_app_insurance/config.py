import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(BASE_DIR, "artifacts", "Linear_model.pickle")
PROJECT_DATA_PATH = os.path.join(BASE_DIR, "artifacts", "project_data.json")

PORT_NUMBER = 8080
HOST = "0.0.0.0"

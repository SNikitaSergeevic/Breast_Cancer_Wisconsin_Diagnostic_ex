import os

# Путь к корню проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пути к директориям проекта
DATA_DIR = os.path.join(BASE_DIR, 'data')                    
LOGS_PATH = os.path.join(BASE_DIR, 'logs')                   
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed')    
RESULTS_DIR = os.path.join(BASE_DIR, 'results')              

# Создаём директории, если они ещё не существуют
for path in [DATA_DIR, LOGS_PATH, PREPROCESSED_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# Словарь с основными конфигурационными путями и параметрами
CONFIG = {
    "raw_data_path": os.path.join(DATA_DIR, "breast_cancer.csv"),
    
    "preprocessed_data_path": os.path.join(PREPROCESSED_DIR, "preprocessed_data.csv"),
    
    "model_path": os.path.join(RESULTS_DIR, "model.pkl"),
    
    "metrics_path": os.path.join(RESULTS_DIR, "metrics.json"),
    
    "target_column": "diagnosis"
}

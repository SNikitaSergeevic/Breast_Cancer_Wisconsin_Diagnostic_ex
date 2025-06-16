import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

# Rорень проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пути для логов и скриптов
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Настройки DAGа
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=30),
    "start_date": datetime(2025, 6, 15),
}

with DAG(
    dag_id="ml_pl",
    description="ML pipeline",
    default_args=default_args,
    catchup=False,
    tags=["pipeline"],
) as dag:
    
    # Загружаем данные
    load_task = BashOperator(
        task_id="load_data",
        bash_command=(
            f"PYTHONPATH={PROJECT_ROOT} python -m etl.load_data >> {os.path.join(LOG_DIR, 'load_data.log')} 2>&1"
        )
    )

    # Предобработка
    preprocess_task = BashOperator(
        task_id="preprocess_data",
        bash_command=(
            f"PYTHONPATH={PROJECT_ROOT} python -m etl.preprocess >> {os.path.join(LOG_DIR, 'preprocess_data.log')} 2>&1"
        )
    )

    # Обучаем модель
    train_task = BashOperator(
        task_id="train_model",
        bash_command=(
            f"PYTHONPATH={PROJECT_ROOT} python -m etl.train_model >> {os.path.join(LOG_DIR, 'train_model.log')} 2>&1"
        )
    )

    # Оценка модели
    evaluate_task = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f"PYTHONPATH={PROJECT_ROOT} python -m etl.evaluate >> {os.path.join(LOG_DIR, 'evaluate_model.log')} 2>&1"
        )
    )

    # Зависимости: поэтапная последовательность
    load_task >> preprocess_task >> train_task >> evaluate_task

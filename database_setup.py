import sqlite3
import pandas as pd
import logging
import os
from pathlib import Path

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = 'housing_data.db'

def init_db() -> None:
    """Initialize the SQLite database schema for predictions."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        logger.info("Initializing database schema...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                longitude REAL,
                latitude REAL,
                housing_median_age REAL,
                total_rooms REAL,
                total_bedrooms REAL,
                population REAL,
                households REAL,
                median_income REAL,
                predicted_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def load_csv_to_db(csv_path: str = 'housing.csv') -> None:
    """Ingests raw CSV data into SQLite to act as the source of truth for training."""
    if not Path(csv_path).exists():
        logger.error(f"Data source {csv_path} not found. Aborting ingestion.")
        return

    try:
        logger.info(f"Ingesting raw data from {csv_path} into SQL...")
        conn = sqlite3.connect(DB_PATH)
        
        df = pd.read_csv(csv_path)
        df.to_sql('training_data', conn, if_exists='replace', index=False)
        
        logger.info(f"Successfully ingested {len(df)} records into 'training_data' table.")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def log_prediction(features: list, prediction: float) -> None:
    """Logs inference requests to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
            population, households, median_income, predicted_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (*features, prediction))
        
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to log prediction: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    logger.info("Starting Data Ingestion Pipeline...")
    init_db()
    load_csv_to_db()
    logger.info("Pipeline execution completed.")

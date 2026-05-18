import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import sqlite3
import logging
from pathlib import Path

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = 'housing_data.db'
MODEL_PATH = 'rf_model.pkl'

def train_pipeline() -> None:
    """Executes the end-to-end training pipeline reading directly from SQL."""
    if not Path(DB_PATH).exists():
        logger.error(f"Database {DB_PATH} not found. Please run data ingestion first.")
        return

    try:
        logger.info("Connecting to SQL database to extract training data...")
        conn = sqlite3.connect(DB_PATH)
        
        # Extract
        df = pd.read_sql_query("SELECT * FROM training_data", conn)
        logger.info(f"Extracted {len(df)} records from database.")
        
        # Transform (Basic Cleaning)
        initial_count = len(df)
        df = df.dropna()
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows containing missing values.")
        
        # Feature Engineering
        features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                    'total_bedrooms', 'population', 'households', 'median_income']
        target = 'median_house_value'
        
        X = df[features]
        y = df[target]
        
        # Model Training
        logger.info("Initiating model training (RandomForestRegressor)...")
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        logger.info("Model training completed successfully.")
        
        # Export Model
        logger.info(f"Serializing and exporting model to {MODEL_PATH}...")
        joblib.dump(model, MODEL_PATH)
        
        logger.info("End-to-End SQL Training Pipeline finished successfully! ✅")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    logger.info("Starting End-to-End Machine Learning Pipeline...")
    train_pipeline()

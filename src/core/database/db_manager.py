"""SQLite database manager for prediction history."""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage SQLite database for prediction history."""
    
    def __init__(self, db_path: Path):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    asd_probability REAL NOT NULL,
                    control_probability REAL NOT NULL,
                    file_path TEXT,
                    epochs_processed INTEGER,
                    model_accuracy REAL,
                    notes TEXT
                )
            """)
            
            # Create app_settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            
            conn.commit()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_prediction(self, results: Dict) -> int:
        """
        Save a prediction to the database.
        
        Args:
            results: Dictionary with prediction results
            
        Returns:
            ID of inserted record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, filename, prediction, confidence,
                    asd_probability, control_probability, file_path,
                    epochs_processed, model_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                results.get('filename', 'Unknown'),
                results.get('prediction', 'Unknown'),
                results.get('confidence', 0.0),
                results.get('asd_probability', 0.0),
                results.get('control_probability', 0.0),
                results.get('file_path', ''),
                results.get('epochs_processed', 0),
                results.get('model_accuracy', 0.0)
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_all_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Get all predictions from database.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of prediction dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """
        Get a specific prediction by ID.
        
        Args:
            prediction_id: ID of prediction to retrieve
            
        Returns:
            Prediction dictionary or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            
            return dict(row) if row else None
    
    def delete_prediction(self, prediction_id: int) -> bool:
        """
        Delete a prediction by ID.
        
        Args:
            prediction_id: ID of prediction to delete
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def clear_all_predictions(self) -> int:
        """
        Delete all predictions.
        
        Returns:
            Number of records deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions")
            conn.commit()
            return cursor.rowcount
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total = cursor.fetchone()[0]
            
            # ASD vs Control counts
            cursor.execute("""
                SELECT prediction, COUNT(*) 
                FROM predictions 
                GROUP BY prediction
            """)
            counts = dict(cursor.fetchall())
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM predictions")
            avg_conf = cursor.fetchone()[0] or 0.0
            
            return {
                'total': total,
                'asd_count': counts.get('ASD DETECTED', 0),
                'control_count': counts.get('NEUROTYPICAL', 0),
                'avg_confidence': avg_conf
            }
    
    def save_setting(self, key: str, value: str) -> None:
        """Save application setting."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
    
    def get_setting(self, key: str, default: str = "") -> str:
        """Get application setting."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default
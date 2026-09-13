import sqlite3
from datetime import datetime

class TrafficDatabase:
    def __init__(self, db_name="traffic_system.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Table for storing traffic analysis & predictions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_vehicles INTEGER,
                avg_speed REAL,
                congestion_level TEXT,
                co2_emission REAL
            )
        ''')
        self.conn.commit()

    def log_traffic_data(self, total_vehicles, avg_speed, congestion_level, co2_emission):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute('''
            INSERT INTO traffic_logs (timestamp, total_vehicles, avg_speed, congestion_level, co2_emission)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, total_vehicles, avg_speed, congestion_level, co2_emission))
        self.conn.commit()

    def get_recent_logs(self, limit=10):
        self.cursor.execute('SELECT timestamp, total_vehicles, avg_speed, congestion_level, co2_emission FROM traffic_logs ORDER BY id DESC LIMIT ?', (limit,))
        return self.cursor.fetchall()

# Testing database locally
if __name__ == "__main__":
    db = TrafficDatabase()
    db.log_traffic_data(245, 22.0, "MODERATE", 2.85)
    print("Database initialized and sample data logged successfully[cite: 1]!")
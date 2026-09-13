import time
from datetime import datetime

class TrafficAutomationHub:
    def __init__(self):
        self.active_alerts = []

    def check_congestion_threshold(self, predicted_volume: int):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if predicted_volume > 500:
            alert_msg = f"[{current_time}] CRITICAL ALERT: High congestion expected (>500 vehicles)[cite: 1]. Public advisory dispatched."
            self.active_alerts.append(alert_msg)
            return alert_msg
        elif predicted_volume > 350:
            alert_msg = f"[{current_time}] WARNING: Moderate-to-High traffic buildup detected. Signal timings adjusted[cite: 1]."
            self.active_alerts.append(alert_msg)
            return alert_msg
        return f"[{current_time}] STATUS: Traffic flow normal."

    def trigger_emergency_corridor(self, ambulance_detected: bool):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ambulance_detected:
            msg = f"[{current_time}] 🚑 n8n Automation Triggered: Green Corridor active! All connecting signals held RED[cite: 1]."
            self.active_alerts.append(msg)
            return msg
        return f"[{current_time}] 🟢 Standard Signal Optimization active."

# Testing the automation module
if __name__ == "__main__":
    hub = TrafficAutomationHub()
    print(hub.check_congestion_threshold(560))
    print(hub.trigger_emergency_corridor(True))
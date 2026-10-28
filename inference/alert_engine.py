from typing import Optional

class AlertEngine:
    """
    Phase 9: Generates early-warning alerts when configurable conditions are satisfied.
    """
    @staticmethod
    def evaluate_and_generate_alert(severity: str, risk_score: float, location_data: dict) -> Optional[str]:
        if severity in ["HIGH", "CRITICAL"] or risk_score > 0.75:
            loc_str = f"{location_data['lat']}, {location_data['lon']}" if location_data.get('available') else "Unknown Location"
            
            alert = (
                "[ALERT] WILDFIRE HAZARD DETECTED\n"
                f"Location: {loc_str}\n"
                f"Severity: {severity}\n"
                f"Risk Score: {risk_score:.2f}\n"
                "Action Required: Immediate review of telemetry data."
            )
            return alert
        return None

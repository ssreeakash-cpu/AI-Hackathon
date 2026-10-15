from typing import Dict, Any

class SeverityEngine:
    """
    Phase 8: Estimates wildfire severity based on actual outputs from models.
    """
    @staticmethod
    def calculate_severity(fire_conf: float, smoke_conf: float, area: float = 0.0, growth_velocity: float = 0.0) -> str:
        # Heavily weight fire confidence, followed by smoke and physical growth
        score = (fire_conf * 0.4) + (smoke_conf * 0.3)
        
        if area > 0:
            score += min(area / 1000.0, 1.0) * 0.2
        if growth_velocity > 0:
            score += min(growth_velocity / 50.0, 1.0) * 0.1
            
        if score > 0.8:
            return "CRITICAL"
        elif score > 0.6:
            return "HIGH"
        elif score > 0.3:
            return "MEDIUM"
        return "LOW"

class RiskFusionEngine:
    """
    Phase 8: Fuses multiple model inputs and environmental data into a final risk score.
    """
    def __init__(self, config: Dict[str, float] = None):
        self.weights = config or {
            "detection": 0.5,
            "temporal": 0.2,
            "growth": 0.2,
            "environmental": 0.1
        }
        
    def calculate_final_risk(self, detection_conf: float, persistence: float, growth_rate: float, env_risk: float) -> float:
        # Normalizing inputs to 0-1 bounds where applicable
        norm_growth = min(growth_rate / 100.0, 1.0)
        
        final_risk = (
            (detection_conf * self.weights["detection"]) +
            (persistence * self.weights["temporal"]) +
            (norm_growth * self.weights["growth"]) +
            (env_risk * self.weights["environmental"])
        )
        return min(max(final_risk, 0.0), 1.0)

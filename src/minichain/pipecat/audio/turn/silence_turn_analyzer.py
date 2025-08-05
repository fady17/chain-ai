# src/minichain/pipecat/audio/turn/silence_turn_analyzer.py
from typing import Any, Dict
import numpy as np

from .smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnParams

class SilenceTurnAnalyzer(BaseSmartTurn):
    """
    A simple but robust Turn Analyzer that uses the fallback silence timer
    from BaseSmartTurn as its primary mechanism. It does not call an ML model.
    """
    def __init__(self, *, sample_rate=None, params: SmartTurnParams = None): # type: ignore
        # We can set a more sensible default for silence-only detection
        if not params:
            params = SmartTurnParams(stop_secs=0.8)
        super().__init__(sample_rate=sample_rate, params=params)

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """
        This analyzer does not use a predictive model. It relies solely on the
        silence timer in the base class. Therefore, it always returns INCOMPLETE,
        letting the silence timer be the only thing that can trigger a COMPLETE state.
        """
        return {"prediction": 0, "probability": 0.0} # Always predict INCOMPLETE
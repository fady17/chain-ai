# src/minichain/pipecat/transcriptions/language.py
import sys
from enum import Enum

# Pipecat uses a custom StrEnum for Python < 3.11 compatibility.
# We will do the same for robustness.
if sys.version_info < (3, 11):
    class StrEnum(str, Enum):
        """String enumeration base class for Python < 3.11 compatibility."""
        def __new__(cls, value):
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj
else:
    from enum import StrEnum


class Language(StrEnum):
    """
    Language codes for speech and text processing services, following BCP 47.
    
    This enum provides standardized language codes to ensure consistency across
    different components like STT, TTS, and LLMs. Egyptian Arabic is treated
    as a first-class citizen.
    """

    # --- Primary Languages ---
    AR_EG = "ar-EG"  # Arabic (Egypt)
    AR_SA = "ar-SA"  # Arabic (Saudi Arabia)
    EN_US = "en-US"  # English (United States)

    # --- Aliases for convenience ---
    EGYPTIAN_ARABIC = "ar-EG"
    SAUDI_ARABIC = "ar-SA"
    US_ENGLISH = "en-US"

    # --- Language-only codes (useful for some services) ---
    AR = "ar"        # Arabic
    EN = "en"        # English

    def __str__(self) -> str:
        return self.value
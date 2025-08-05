
# src/minichain/voice/azure_tts.py
from .base import BaseTTSService, AzureTTSConfig

class AzureTTSService(BaseTTSService):
    def __init__(self, config: AzureTTSConfig):
        super().__init__(config)
    
    def get_pipecat_service(self):
        try:
            from pipecat.services.azure.tts import AzureTTSService as PipecatAzureTTS
        except ImportError:
            raise ImportError("Pipecat dependencies not found. Run `pip install 'minichain-ai[voice]'`.")

        config = self.config
        assert isinstance(config, AzureTTSConfig)
        print(f"🔧 DEBUG: Creating Pipecat TTS with voice: {config.voice}") #delete
        return PipecatAzureTTS(api_key=config.api_key, region=config.region, voice_name=config.voice)
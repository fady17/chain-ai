# src/minichain/voice/azure_stt.py
from .base import BaseSTTService, AzureSTTConfig

class AzureSTTService(BaseSTTService):
    def __init__(self, config: AzureSTTConfig):
        super().__init__(config)

    def get_pipecat_service(self):
        try:
            from pipecat.services.azure.stt import AzureSTTService as PipecatAzureSTT
        except ImportError:
            raise ImportError("Pipecat dependencies not found. Run `pip install 'minichain-ai[voice]'`.")
        
        config = self.config
        assert isinstance(config, AzureSTTConfig)
        return PipecatAzureSTT(api_key=config.api_key, region=config.region)
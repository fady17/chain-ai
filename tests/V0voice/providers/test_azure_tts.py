# # tests/voice/providers/test_azure_tts.py
# """
# Tests for the `minichain.voice.providers.tts.AzureTTSService` wrapper.
# """
# import os
# import sys
# import pytest
# from dotenv import load_dotenv

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

# @pytest.fixture(scope="module")
# def pipecat_azure_tts_class():
#     """Fixture to safely import the underlying Pipecat class, skipping if unavailable."""
#     try:
#         from pipecat.services.azure.tts import AzureTTSService as PipecatClass
#         return PipecatClass
#     except ImportError:
#         pytest.skip("Skipping TTS tests: `pipecat-ai` or its dependencies not installed.")

# from minichain.voice.providers.tts import AzureTTSService
# from minichain.voice.base import AzureTTSConfig

# load_dotenv()

# def test_wrapper_initialization(pipecat_azure_tts_class):
#     """Tests that the wrapper class instantiates correctly with a config object."""
#     config = AzureTTSConfig(api_key="dummy", region="dummy", voice="dummy")
#     service = AzureTTSService(config=config)
#     assert service.config.voice == "dummy"

# def test_get_pipecat_service_returns_correct_type(pipecat_azure_tts_class):
#     """
#     Tests that the factory method returns a correctly configured instance of the
#     underlying Pipecat service.
#     """
#     api_key = os.getenv("AZURE_SPEECH_KEY")
#     region = os.getenv("AZURE_SPEECH_REGION")
#     if not api_key or not region:
#         pytest.skip("Skipping TTS integration test: Azure Speech credentials not found.")
    
#     config = AzureTTSConfig(api_key=api_key, region=region, voice="en-US-JennyNeural")
#     wrapper = AzureTTSService(config=config)
    
#     pipecat_instance = wrapper.get_pipecat_service()
    
#     assert isinstance(pipecat_instance, pipecat_azure_tts_class)
#     assert pipecat_instance._voice_id == "en-US-JennyNeural"
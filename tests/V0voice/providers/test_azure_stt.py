# # tests/voice/providers/stt/test_azure_stt.py

# import pytest
# from dotenv import load_dotenv

# # --- Correct Path Setup ---
# # This robustly finds the project root and adds the 'src' directory.
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# @pytest.fixture(scope="module")
# def pipecat_azure_stt_class():
#     try:
#         from pipecat.services.azure.stt import AzureSTTService as PipecatClass
#         return PipecatClass
#     except ImportError:
#         pytest.skip("Skipping STT tests: `pipecat-ai` not installed.")

# from minichain.voice.providers.stt import AzureSTTService
# from minichain.voice.base import AzureSTTConfig

# load_dotenv()

# @pytest.mark.usefixtures("pipecat_azure_stt_class")
# def test_wrapper_initialization():
#     config = AzureSTTConfig(api_key="dummy_key", region="dummy_region")
#     AzureSTTService(config=config)

# @pytest.mark.usefixtures("pipecat_azure_stt_class")
# def test_get_pipecat_service_returns_correct_type(pipecat_azure_stt_class):
#     api_key = os.getenv("AZURE_SPEECH_KEY")
#     region = os.getenv("AZURE_SPEECH_REGION")
#     if not api_key or not region:
#         pytest.skip("Skipping STT integration test: Azure Speech credentials not found.")
    
#     config = AzureSTTConfig(api_key=api_key, region=region)
#     wrapper = AzureSTTService(config=config)
#     pipecat_instance = wrapper.get_pipecat_service()
    
#     assert isinstance(pipecat_instance, pipecat_azure_stt_class)
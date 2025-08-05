# # tests/voice/services/test_pipecat_service.py
# """
# Tests for the high-level `PipecatVoiceService` orchestrator.
# """
# import sys
# import os
# import pytest
# from unittest.mock import patch, MagicMock, AsyncMock

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# from minichain.voice.services import PipecatVoiceService
# from minichain.voice.base import BaseSTTService, BaseTTSService
# from minichain.chat_models.base import BaseChatModel

# @pytest.fixture(scope="module")
# def check_pipecat_installed():
#     """Skips tests in this file if pipecat-ai is not installed."""
#     try:
#         import pipecat
#     except ImportError:
#         pytest.skip("Skipping voice service tests: `pipecat-ai` not installed.")

# # --- Mock Implementations ---

# class MockSTTService(BaseSTTService):
#     def __init__(self): pass # Dummy init
#     def get_pipecat_service(self) -> MagicMock:
#         return MagicMock()

# class MockTTSService(BaseTTSService):
#     def __init__(self): pass # Dummy init
#     def get_pipecat_service(self) -> MagicMock:
#         return MagicMock()

# class MockChatModel(BaseChatModel):
#     def __init__(self): pass # Dummy init
#     def invoke(self, input_data) -> str:
#         return "mocked response"
#     def stream(self, input_data):
#         yield "mocked"
#         yield " response"

# # --- Test Functions ---

# @pytest.mark.usefixtures("check_pipecat_installed")
# @patch('minichain.voice.services.pipecat_service.PipelineRunner')
# def test_service_assembles_and_runs_pipeline(MockPipelineRunner):
#     """
#     Tests that `PipecatVoiceService.run()` correctly assembles all necessary
#     Pipecat components (transport, STT, TTS, bridge) and attempts to start
#     the runner. The runner itself is mocked to prevent blocking.
#     """
#     # ARRANGE
#     mock_runner_instance = MockPipelineRunner.return_value
#     mock_runner_instance.run = AsyncMock()  # Make the run method awaitable

#     mock_stt = MockSTTService()
#     mock_tts = MockTTSService()
#     mock_model = MockChatModel()
    
#     # Create the service instance with our mock components
#     voice_service = PipecatVoiceService(
#         model=mock_model,
#         stt_service=mock_stt,
#         tts_service=mock_tts,
#         system_prompt="test"
#     )
    
#     # ACT
#     try:
#         voice_service.run()
#     except SystemExit:
#         pass # Ignore potential exits in test environment

#     # ASSERT
#     # The most important assertion: was the .run() method of the runner called?
#     # This proves all the setup logic before it was successful.
#     mock_runner_instance.run.assert_awaited_once()

#     # A secondary check: were the provider's factory methods called?
#     mock_stt.get_pipecat_service.assert_called_once()
#     mock_tts.get_pipecat_service.assert_called_once()
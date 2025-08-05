# # src/minichain/voice/vad/silero.py

# # --- Optional Import Handling ---
# # This block allows the module to be imported, but using SileroVAD will fail
# # at runtime if torch/torchaudio are not installed.
# try:
#     import torch
#     import torchaudio
#     _torch_available = True
# except ImportError:
#     _torch_available = False
# # ---

# from minichain.voice.vad.base import BaseVAD
# from minichain.voice.audio import RATE

# class SileroVAD(BaseVAD):
#     """
#     A deep-learning-based Voice Activity Detector powered by the Silero VAD model.
#     This VAD is more accurate and robust to noise than WebRtcVAD, but has
#     heavier dependencies (PyTorch).

#     This class is stateful and must be used for a single, continuous audio stream.
#     """
#     def __init__(self, threshold: float = 0.5):
#         """
#         Initializes the Silero VAD model. This will download the model on first use.

#         Args:
#             threshold (float): The confidence threshold (0.0 to 1.0) for classifying
#                 a chunk as speech. Defaults to 0.5.
#         """
#         if not _torch_available:
#             raise ImportError(
#                 "Torch and Torchaudio are not installed. "
#                 "Please install them to use SileroVAD: `pip install torch torchaudio`"
#             )

#         if not 0.0 <= threshold <= 1.0:
#             raise ValueError("Threshold must be between 0.0 and 1.0.")
        
#         self.threshold = threshold
        
#         # Use a local cache for the model to avoid re-downloading
#         torch.set_num_threads(1) # Optimize for single-stream inference
#         self.model, self.utils = torch.hub.load(
#             repo_or_dir='snakers4/silero-vad',
#             model='silero_vad',
#             force_reload=False # Use cached model if available
#         )
        
#         # Silero VAD is stateful; it uses a hidden state (_h) and context (_c)
#         # to track speech across chunks. We manage this state within the class.
#         self._h = None
#         self._c = None
        
#     def _prepare_audio_tensor(self, chunk: bytes) -> "torch.Tensor":
#         """Converts a raw 16-bit PCM byte chunk to a float32 PyTorch tensor."""
#         # Convert byte string to a tensor of 16-bit integers
#         audio_int16 = torch.frombuffer(chunk, dtype=torch.int16)
#         # Convert to a float tensor in the range [-1.0, 1.0], required by the model
#         audio_float32 = audio_int16.to(torch.float32) / 32768.0
#         return audio_float32

#     def is_speech(self, chunk: bytes) -> bool:
#         """
#         Uses the stateful Silero VAD model to determine if an audio chunk contains speech.

#         Args:
#             chunk: A raw audio chunk (16-bit PCM). The length is not as strict as
#                    WebRtcVAD, but should be consistent.

#         Returns:
#             bool: True if speech probability is above the threshold, False otherwise.
#         """
#         if not isinstance(chunk, bytes):
#             raise TypeError("Audio chunk must be of type bytes.")

#         audio_tensor = self._prepare_audio_tensor(chunk)
            
#         # The model updates its internal state and returns the speech probability.
#         # We pass the hidden state (_h, _c) from the previous call.
#         speech_prob, new_state = self.model(audio_tensor, RATE, h=self._h, c=self._c)
        
#         # Update the state for the next chunk
#         self._h = new_state.get('h')
#         self._c = new_state.get('c')
        
#         return speech_prob.item() > self.threshold

#     def reset_state(self):
#         """Resets the VAD's internal state. Call this before processing a new stream."""
#         self.model.reset_states()
#         self._h = None
#         self._c = None
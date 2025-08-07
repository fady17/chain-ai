import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.chat_models.base import LocalChatConfig
from minichain.chat_models.local import LocalChatModel

# No special flags needed. This will use the fast non-streaming API.
# config = OpenRouterConfig(model_name="mistralai/mistral-7b-instruct")
# model = OpenRouterChatModel(config=config)
model = LocalChatModel(config=LocalChatConfig())

# This is fast and clean.
response = print(model.invoke("Who is the strongest Hashira?"))
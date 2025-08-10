import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.chat_models import OpenRouterConfig, OpenRouterChatModel


# The user explicitly tells the library how to handle this model.
config_defaults = OpenRouterConfig(model="qwen/qwen3-235b-a22b:free") # type: ignore
model_defaults = OpenRouterChatModel(config=config_defaults , reasoning={"effort": "high", "exclude": True})
# This will now use the stream-and-clean method and return the correct answer.
response =model_defaults.invoke("what's your name?")
print(response) # Output: My name is Qwen.
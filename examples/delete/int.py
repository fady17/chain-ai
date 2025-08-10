import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.chat_models import LocalChatModel, LocalChatConfig,OpenRouterConfig,OpenRouterChatModel, AzureOpenAIChatModel, AzureChatConfig

# One-liner for LocalChatModel
local_model = LocalChatModel(config=LocalChatConfig())

model = OpenRouterChatModel(config=OpenRouterConfig(model="qwen/qwen3-235b-a22b:free", temperature=0))

print("qwenLocal: " +local_model.invoke("what's your name"))

print("OpenRouterQwen: " +model.invoke("what's your name ?"))

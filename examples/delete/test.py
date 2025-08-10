# from IPython.display import display, Markdown
# def display_markdown(text: str):
#     display(Markdown(text))
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.chat_models import OpenRouterConfig, OpenRouterChatModel

config = OpenRouterConfig(
    model="qwen/qwen3-235b-a22b:free",  # Or any model from OpenRouter
)

llm = OpenRouterChatModel(config=config)
text = llm.invoke("Who is the strongest Hashira ?")
print(text)

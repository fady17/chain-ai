# Make sure to set the environment variable or pass the key directly
# export OPENROUTER_API_KEY="sk-or-..."
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.prompts import PromptTemplate
from chain.chat_models import OpenRouterConfig, OpenRouterChatModel

# --- Setup ---
config = OpenRouterConfig(model="qwen/qwen3-235b-a22b:free")
llm = OpenRouterChatModel(config=config)
prompt_template = PromptTemplate.from_template("Tell me a joke about {{topic}}")
prompt_value = prompt_template.invoke({"topic": "cats"})


# --- OLD WAY (STILL WORKS PERFECTLY) ---
print("--- Using invoke (returns a string) ---")
response_str = llm.invoke(prompt_value)
print(type(response_str))
print(response_str)


# --- NEW, POWERFUL WAY ---
print("\n--- Using generate (returns a ChatResult object) ---")
result_obj = llm.generate(prompt_value)
print(type(result_obj))

# Access the rich metadata
print(f"Content: {result_obj.content}")
print(f"Model Used: {result_obj.model_name}")
print(f"Finish Reason: {result_obj.finish_reason}")
print(f"Tokens Used: {result_obj.token_usage.total_tokens}")

# The object still prints nicely thanks to __str__
print("\nPrinting the object directly:")
print(result_obj)

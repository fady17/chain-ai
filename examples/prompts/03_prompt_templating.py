# examples/03_prompt_templating.py
"""
Example 3: A tour of the Mini-Chain prompt templating engine.

This script demonstrates all three core prompt templates:
- PromptTemplate: For simple, powerful string formatting with Jinja2.
- FewShotPromptTemplate: For teaching the model a task with examples.
- ChatPromptTemplate: For structuring conversations for chat models.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate

def print_header(title): print(f"\n--- {title} ---")

# 1. PromptTemplate (The Workhorse)
print_header("1. PromptTemplate with Jinja2 Logic")
pt = PromptTemplate(
    template="Report for {{ user }}:\nItems:\n{% for item in items %}- {{ item }}\n{% endfor %}"
)
result_pt = pt.format(user="Amin", items=["Task 1", "Task 2"])
print(result_pt)

# 2. FewShotPromptTemplate (The Teacher)
print_header("2. FewShotPromptTemplate for Classification")
examples = [{"query": "Help me with my code.", "intent": "Technical"}]
example_pt = PromptTemplate(template="Query: {{ query }}\nIntent: {{ intent }}")
fs_pt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_pt,
    suffix="Query: {{ user_input }}\nIntent:",
    input_variables=["user_input"]
)
result_fs = fs_pt.format(user_input="This library is great!")
print(result_fs)

# 3. ChatPromptTemplate (The Conversationalist)
print_header("3. ChatPromptTemplate for Personas")
chat_pt = ChatPromptTemplate(
    messages=[
        {"role": "system", "content": "You are a helpful {{ persona }}."},
        {"role": "user", "content": "Tell me a joke about {{ topic }}."}
    ]
)
result_chat = chat_pt.format(persona="pirate", topic="computers")
import json
print(json.dumps(result_chat, indent=2))
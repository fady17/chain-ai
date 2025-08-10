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

from chain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate

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

# 1. Define the example prompt in LangChain format
example_prompt = PromptTemplate(
    template="Question: {{ question }}\n{{ answer }}"
)
# 2. Define the examples list
examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]

# 3. Create the FewShotPromptTemplate instance
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {{ input }}",
    input_variables=["input"]
)

# 4. Invoke with input and convert to string
final_prompt = prompt.invoke({"input": "Who was the father of Killua Zoldyck?"}).to_string()

# 5. Print the result
print(final_prompt)
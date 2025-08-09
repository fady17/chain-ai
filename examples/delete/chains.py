import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.prompts import PromptTemplate
from minichain.chat_models import LocalChatConfig, LocalChatModel
from minichain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- 1. Define the desired output structure ---
class Joke(BaseModel):
    """A structured representation of a joke."""
    setup: str = Field(description="The setup or question part of the joke.")
    punchline: str = Field(description="The punchline or answer to the joke.")

# --- 2. Create the individual components ---
parser = PydanticOutputParser(pydantic_object=Joke)

# Use correct Jinja2 syntax and partial_variables
prompt = PromptTemplate(
    template="Answer the user query.\n{{ format_instructions }}\n{{ query }}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

config = LocalChatConfig() 
llm = LocalChatModel(config=config)

# --- 3. Create the chain using the `|` operator ---
# This will now work because the base classes have been patched
chain = prompt | llm | parser # type: ignore

# --- 4. Invoke the entire chain ---
print("Invoking the chain...")
joke_query = "Tell me a joke about a programmer."
result = chain.invoke({"query": joke_query})

# --- 5. Inspect the result ---
print("\n--- Chain Result ---")
print(f"Type of result: {type(result)}")
print(f"Setup: {result.setup}")
print(f"Punchline: {result.punchline}")
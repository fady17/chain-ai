import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from chain.prompts import PromptTemplate
from chain.chat_models import LocalChatConfig, LocalChatModel
from chain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

def main():
    class Joke(BaseModel):
        setup: str = Field(description="The setup or question part of the joke.")
        punchline: str = Field(description="The punchline or answer to the joke.")

    parser = PydanticOutputParser(pydantic_object=Joke)

    prompt = PromptTemplate(
        template="Answer the user query.\n{{ format_instructions }}\n{{ query }}\n",
        input_variables=["query"], # This correctly tells the chain what to expect
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    config = LocalChatConfig()
    llm = LocalChatModel(config=config)
    chain = prompt | llm | parser # type: ignore

    print("Invoking the chain...")
    joke_query = "Tell me a joke about a programmer."
    
    # This call now works correctly:
    # 1. chain.invoke passes {"query": joke_query} to prompt.invoke.
    # 2. prompt.invoke calls prompt.format(**{"query": joke_query}).
    # 3. prompt.format combines it with partial_variables and renders the template.
    # 4. The fully rendered string is passed to the LLM.
    result = chain.invoke({"query": joke_query})

    print("\n--- Chain Result ---")
    print(result)

if __name__ == "__main__":
    main()
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.chat_models import LocalChatModel, LocalChatConfig
# from minichain.text_splitters import RecursiveCharacterTextSplitter
from minichain.text_splitters import RecursiveCharacterTextSplitter



locale_config = LocalChatConfig(model="qwen2.5-7b-instruct")
local_model = LocalChatModel(config=locale_config)
# Load example document
with open("/Users/fady/Desktop/internship/langchain-clone/chainforge-ai/examples/delete/datasets/story.txt") as f:
    state_of_the_union = f.read()
llm = local_model
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    # separators=["asdfasdfasdf"],
)
texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print("-------------------")
print(texts[1])
print("-------------------")
print(texts[2])
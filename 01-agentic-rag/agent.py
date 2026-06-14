"""Q6: Agentic RAG with toyaikit.

Wires a `search` tool over the chunk index from Q4/Q5, registers it as a
function-calling tool, and lets the LLM decide when (and how often) to invoke
it. Counts how many times the tool gets called for the homework question.
"""

import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI

from toyaikit.tools import Tools
from toyaikit.llm import OpenAIChatCompletionsClient
from toyaikit.chat.runners import OpenAIChatCompletionsRunner
from toyaikit.chat import IPythonChatInterface

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index


# ---------- 1. Build the chunk index (same as Q4/Q5) ----------

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
files = reader.read()
documents = [f.parse() for f in files]
chunks = chunk_documents(documents, size=2000, step=1000)

chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)


# ---------- 2. The search tool ----------

# We wrap it in a class so we can hold a counter and the index.
# toyaikit auto-builds the tool schema from the type hints and docstring,
# so the docstring style matters — keep it descriptive.

class SearchTools:
    def __init__(self, index):
        self.index = index
        self.call_count = 0

    def search(self, query: str) -> list[dict]:
        """
        Search the course lesson chunks for passages relevant to the query.

        Args:
            query (str): The search query — keywords or a short question.

        Returns:
            list[dict]: A list of matching chunks, each with `filename` and `content`.
        """
        self.call_count += 1
        return self.index.search(query, num_results=5)


search_tools = SearchTools(chunk_index)

tools = Tools()
tools.add_tools(search_tools)


# ---------- 3. The agent ----------

load_dotenv()

raw_openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    http_client=httpx.Client(verify=False),
)

llm_client = OpenAIChatCompletionsClient(
    model="devstral",
    client=raw_openai_client,
)

DEVELOPER_PROMPT = """
You're a course teaching assistant. Answer the student's question using the
search tool. Make multiple searches with different keywords before answering.
""".strip()

runner = OpenAIChatCompletionsRunner(
    tools=tools,
    developer_prompt=DEVELOPER_PROMPT,
    chat_interface=IPythonChatInterface(),
    llm_client=llm_client,
)


# ---------- 4. Run it on the homework question ----------

QUESTION = "How does the agentic loop work, and how is it different from plain RAG?"

# Reset counter and run a single loop turn (no interactive prompt).
search_tools.call_count = 0
result = runner.loop(prompt=QUESTION)

print()
print("=" * 60)
print(f"Search tool was called: {search_tools.call_count} times")
print("=" * 60)
print()
print("Final answer:")
print(result.last_message)
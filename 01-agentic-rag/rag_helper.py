"""RAG helper for the LLM-Zoomcamp homework.

The RAG class wires together a minsearch index and an OpenAI-compatible client.
`rag(question)` returns BOTH the model's answer and the token-usage object, so
the homework can read input (prompt) tokens from it.
"""

from dataclasses import dataclass
from typing import Any


INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.
Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
""".strip()


USER_PROMPT_TEMPLATE = """
Question: {question}
 
Context:
{context}
""".strip()


@dataclass
class RAGResult:
	answer: str
	usage: Any


class RAG:
	def __init__(self, index, openai_client, model: str = "devstral"):
		self.index = index
		self.client = openai_client
		self.model = model

	def search(self, query: str, num_results: int = 5) -> list[dict]:
		return self.index.search(query, num_results=num_results)

	def build_context(self, search_results: list[dict]) -> str:
		lines = []
		for doc in search_results:
			lines.append(f"filename: {doc['filename']}")
			lines.append(doc["content"])
			lines.append("")
		return "\n".join(lines).strip()

	def build_prompt(self, question: str, search_results: list[dict]) -> str:
		context = self.build_context(search_results)
		return USER_PROMPT_TEMPLATE.format(question=question, context=context)

	def llm(self, user_prompt: str):
		"""Call the model and return the FULL response (not just the text).

		Returning the whole response lets the caller read usage info.
		"""
		response = self.client.chat.completions.create(
			model=self.model,
			messages=[
				{"role": "system", "content": INSTRUCTIONS},
				{"role": "user", "content": user_prompt},
			],
		)
		return response

	def rag(self, question: str, num_results: int = 5) -> RAGResult:
		search_results = self.search(question, num_results=num_results)
		user_prompt = self.build_prompt(question, search_results)
		response = self.llm(user_prompt)
		answer = response.choices[0].message.content
		return RAGResult(answer=answer, usage=response.usage)

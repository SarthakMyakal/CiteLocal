"""Test doubles so tests never need Ollama running."""
from langchain_core.messages import AIMessage


class FakeLLM:
    """Returns scripted replies in order. Replies may be strings or AIMessages (for tool calls)."""

    def __init__(self, replies: list, default: str = "YES"):
        self.replies = list(replies)
        self.default = default
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else self.default
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)

    def bind_tools(self, tools):
        return self


class FakeRetriever:
    """Always returns the same chunks."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    def retrieve(self, query, doc_ids=None, collection=None, **kwargs):
        return [dict(c) for c in self.chunks]

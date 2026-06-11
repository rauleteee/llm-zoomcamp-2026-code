
"""Data loading and index building for the LLM-Zoomcamp homework."""
 
from gitsource import GithubRepositoryDataReader
from minsearch import Index
 
 
def load_data():
    """Download lesson markdown files from the course repo at a pinned commit."""
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    files = reader.read()
    documents = [file.parse() for file in files]
    return documents
 
 
def build_index(documents):
    """Index documents with `content` as text and `filename` as keyword."""
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    index.fit(documents)
    return index
 
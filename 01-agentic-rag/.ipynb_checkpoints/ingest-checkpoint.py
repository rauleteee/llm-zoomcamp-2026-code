import requests
from minsearch import Index

def load_data():
    from gitsource import GithubRepositoryDataReader
    
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    
    return files

def build_index(documents):
    from minsearch import Index
    docs = [{"filename": f.filename, "content": f.content} for f in files]
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(docs)
    
    return index
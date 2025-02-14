from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound
import sys

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Embed a list of texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        """Embed a single text."""
        embedding = self.model.encode([text])
        return embedding[0].tolist()  # Return the first embedding since we only encoded one text 

def format_code_for_display(code_snippet: str, language: str = None, line_start: int = 1) -> str:
    """Format code snippet with syntax highlighting and line numbers"""
    # Force PHP lexer and ensure it recognizes PHP code
    lexer = get_lexer_by_name('php', stripall=True)
    lexer.startinline = True  # Treat code as if it's already inside <?php

    highlighted = highlight(
        code_snippet,
        lexer,
        HtmlFormatter(
            linenos=True,
            linenostart=line_start,
            cssclass="source",
            style="monokai"
        )
    )
    return highlighted 
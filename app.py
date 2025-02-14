from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import logging
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Starting the application...')

try:
    from langchain_community.vectorstores import Qdrant
except ImportError as e:
    logger.error(f"ImportError: {e}")
    exit(1)

from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

# Set OpenAI API key as environment variable (recommended)
# Or, for testing only: os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

repo_path = "/app/repo"
all_code = ""

try:
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):  # Or other relevant extensions
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        code = f.read()
                        all_code += code + "\n\n"
                except Exception as e:
                    logger.error(f"Error reading file {filepath}: {e}")
except Exception as e:
    logger.error(f"Error reading repository: {e}")

def embed_documents(documents):
    return model.encode(documents, show_progress_bar=False)

try:
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""]
    )
except Exception as e:
    logger.error(f"Error initializing embeddings or text splitter: {e}")
    exit(1)

try:
    texts = text_splitter.split_text(all_code)
    docsearch = Qdrant.from_texts(texts, embed_documents, location=':memory:')
except Exception as e:
    logger.error(f"Error creating Qdrant index: {e}")
    exit(1)

logger.info('Qdrant index created successfully.')

logger.info('Finished initializing the application...')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query = request.form['query']
    try:
        query_embedding = model.encode(query, show_progress_bar=False)
        docs = docsearch.search(query_embedding, limit=10)
        results = [doc.page_content for doc in docs]
        return render_template('results.html', query=query, results=results)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "Error processing query", 500

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error running application: {e}")

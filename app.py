from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
import logging
import os
from src.embeddings import SentenceTransformerEmbeddings, format_code_for_display
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from qdrant_client import QdrantClient
from threading import Thread

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Starting the application...')

repo_path = "/app/repo"

# Directories to skip
SKIP_DIRS = {
    # Package managers and dependencies
    'node_modules',
    'vendor',
    'bower_components',
    'packages',
    'composer',
    'yarn',
    'npm',
    
    # Build outputs and caches
    'dist',
    'build',
    '__pycache__',
    '.cache',
    'tmp',
    'temp',
    
    # Version control
    '.git',
    '.svn',
    
    # Documentation
    'docs',
    'doc',
    'documentation',
    
    # Test directories
    'tests',
    'test',
    'testing',
    '__tests__',
    
    # IDE and editor files
    '.idea',
    '.vscode',
    '.vs',
    
    # Asset directories (if you want to skip these)
    'assets',
    'images',
    'img',
    'fonts',
    'videos',
    'media'
}

# File patterns to look for
RELEVANT_FILE_PATTERNS = [
    'auth', 'session', 'login', 'user', 'jwt', 'token',  # Authentication related
    'middleware', 'routes', 'views', 'controllers',  # Web framework files
    'security', 'permissions',  # Security related
    'app', 'main', 'index',  # Main application files
    'flask_jwt', 'flask_login',  # Flask specific
    'auth.py', 'authentication.py',  # Common auth file names
    'api', 'rest', 'graphql',  # API related
    '.py', '.js', '.ts'  # Include all Python/JS/TS files as potentially relevant
]

# Paths to look for
RELEVANT_PATHS = [
    'auth',
    'middleware',
    'routes',
    'api',
    'controllers',
    'views',
    'src',
    'app',
    'lib',
    'core',
    'backend',
    'server'
]

# Near the top with other globals
indexing_progress = {
    'current': 0,
    'total': 0,
    'status': 'idle',  # idle, processing_files, creating_vectors, complete, error
    'message': '',
    'phase': 'idle',  # idle, files, vectors, complete
    'indexed_files': []  # Add this to track files
}

# Add near the top with other globals
VECTOR_MESSAGES = [
    'Creating vector index... This may take a few minutes.',
    'Creating vector index... Please wait, still indexing',
    'Creating vector index... Don\'t leave this page',
    'Creating vector index... Your patience is appreciated'
]

# Add debug logging for directory skipping
def should_skip_directory(dirname):
    should_skip = dirname.lower() in SKIP_DIRS or 'cache' in dirname.lower()
    if should_skip:
        logger.debug(f"Skipping directory: {dirname}")
    return should_skip

# Move all the indexing code into a function
def initialize_search():
    global indexing_progress
    try:
        logger.info("Starting indexing process...")
        indexing_progress['status'] = 'indexing'
        indexing_progress['phase'] = 'files'  # Explicitly set phase
        logger.info(f"Current phase: {indexing_progress['phase']}")
        embeddings = SentenceTransformerEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # First pass to count files
        indexing_progress['message'] = 'Counting files...'
        total_files = 0
        logger.info(f"Starting file scan from: {repo_path}")
        for root, dirs, files in os.walk(repo_path):
            # Log directories being considered
            logger.info(f"Scanning directory: {root}")
            logger.info(f"Found directories: {', '.join(dirs)}")
            
            # Skip unwanted directories
            original_dirs = dirs.copy()
            dirs[:] = [d for d in dirs if not should_skip_directory(d)]
            if len(dirs) != len(original_dirs):
                logger.info(f"Filtered directories from {len(original_dirs)} to {len(dirs)}")
            
            # Log files being counted
            valid_files = [f for f in files if f.endswith(('.php', '.js', '.ts', '.html', '.twig', '.py', '.yaml', '.yml', '.json'))]
            if valid_files:
                logger.info(f"Found files in {root}: {', '.join(valid_files)}")
            total_files += len(valid_files)

        logger.info(f"Total files to process: {total_files}")
        
        indexing_progress['total'] = total_files
        indexing_progress['current'] = 0
        indexing_progress['message'] = 'Processing files...'

        texts = []
        metadatas = []
        
        # Process files with progress
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not should_skip_directory(d)]
            
            for file in files:
                if file.endswith(('.php', '.js', '.ts', '.html', '.twig', '.py', '.yaml', '.yml', '.json')):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, repo_path)
                    
                    indexing_progress['message'] = f'Processing {rel_path}...'
                    indexing_progress['current'] += 1
                    
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            code = f.read()
                            chunks = text_splitter.split_text(code)
                            # Calculate line numbers for each chunk
                            line_numbers = []
                            current_line = 1
                            for chunk in chunks:
                                chunk_lines = chunk.count('\n') + 1
                                line_numbers.append({
                                    'start': current_line,
                                    'end': current_line + chunk_lines - 1
                                })
                                logger.info(f"Chunk from {current_line} to {current_line + chunk_lines - 1}")
                                logger.info(f"Content: {chunk[:50]}...")
                                current_line += chunk_lines

                            texts.extend(chunks)
                            metadatas.extend([{
                                "source": rel_path,
                                "line_start": line_nums['start'],
                                "line_end": line_nums['end']
                            } for line_nums in line_numbers])
                            logger.info(f"Processed {rel_path}: {len(chunks)} chunks")
                            indexing_progress['indexed_files'].append(rel_path)
                    except Exception as e:
                        logger.error(f"Error reading file {rel_path}: {e}")

        logger.info("File processing complete, starting vector creation...")
        indexing_progress.update({
            'phase': 'vectors',
            'message': VECTOR_MESSAGES[0],  # Start with first message
            'message_index': 0,  # Add message index to track rotation
            'message_count': 0,  # Add counter for message rotation
            'current': 0,
            'total': len(texts)
        })
        
        result = Qdrant.from_texts(
            texts,
            embeddings,
            metadatas=metadatas,
            location="http://vectorstore:6333",
            collection_name="code_chunks",
            prefer_grpc=False,
            timeout=60
        )
        
        return result
        
    except Exception as e:
        indexing_progress['status'] = 'error'
        indexing_progress['message'] = str(e)
        raise

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'  # Make sure static folder is configured
# Set a secret key for session management
app.secret_key = os.urandom(24)  # Or use a fixed secret key if you prefer

# Global variable for the search index
docsearch = None

@app.route('/')
def index():
    return render_template('index.html', docsearch=docsearch is not None)

@app.route('/query', methods=['GET'])
def query():
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('index'))
    if docsearch is None:
        flash("No index available. Please index your codebase first.", "error")
        return redirect(url_for('index'))
    try:
        enhanced_query = f"{query} jwt token authentication"
        docs = docsearch.similarity_search_with_score(enhanced_query, k=10)
        logger.info(f"Found {len(docs)} initial results")
        
        # Prepare results with more structure
        results = []
        seen = set()
        for doc, score in docs:
            content_lower = doc.page_content.lower()
            if ('jwt' in content_lower or 'token' in content_lower) and score < 0.3:
                content = doc.page_content.strip()
                if content not in seen:
                    seen.add(content)
                    formatted_content = format_code_for_display(
                        content, 
                        'php',
                        line_start=doc.metadata.get('line_start', 1)
                    )
                    results.append({
                        'filepath': doc.metadata.get('source', 'unknown'),
                        'content': formatted_content,
                        'line_start': doc.metadata.get('line_start', 1),
                        'line_end': doc.metadata.get('line_end', 1),
                        'score': score
                    })
        
        logger.info(f"Reduced to {len(results)} unique results")
        
        if not results:
            results = [{
                'filepath': 'No results',
                'content': 'No relevant code found for your query.',
                'score': 0
            }]
            
        return render_template('results.html', query=query, results=results[:5])
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(f"Error details: {str(e)}")
        return "Error processing query", 500

def collection_exists():
    try:
        # Try to get collection info - will raise exception if doesn't exist
        response = requests.get("http://vectorstore:6333/collections/code_chunks")
        logger.info(f"Collection check response: {response.status_code}")
        logger.info(f"Collection info: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        return False

def clear_index():
    try:
        response = requests.delete("http://vectorstore:6333/collections/code_chunks")
        logger.info(f"Clear index response: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        return False

def background_reindex():
    global docsearch, indexing_progress
    try:
        # Reset everything at start
        indexing_progress.update({
            'phase': 'files',
            'status': 'processing',
            'message': 'Processing files...',
            'current': 0,
            'total': 0,
            'indexed_files': []
        })
        
        # initialize_search will handle the file processing phase
        docsearch = initialize_search()
        
        # Mark as complete
        indexing_progress.update({
            'phase': 'complete',
            'status': 'complete',
            'message': 'Indexing complete'
        })
        
    except Exception as e:
        logger.error(f"Error during background reindex: {e}")
        indexing_progress.update({
            'status': 'error',
            'phase': 'error',
            'message': str(e)
        })

@app.route('/admin/reindex', methods=['POST'])
def force_reindex():
    try:
        clear_index()
        # Reset progress
        global indexing_progress
        indexing_progress['current'] = 0
        indexing_progress['total'] = 0
        indexing_progress['status'] = 'idle'
        indexing_progress['message'] = 'Starting...'
        
        # Start indexing in background
        Thread(target=background_reindex).start()
        
        return redirect(url_for('show_progress'))
    except Exception as e:
        logger.error(f"Error during reindex: {e}")
        flash(f"Error during reindex: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/admin/clear', methods=['POST'])
def admin_clear_index():
    global docsearch
    try:
        if clear_index():
            docsearch = None
            flash("Index cleared successfully", "success")
        else:
            flash("Error clearing index", "error")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        flash(f"Error clearing index: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/admin/progress')
def show_progress():
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
              request.headers.get('Accept') == 'application/json'
    
    if is_ajax:
        # Rotate message if in vector phase, but only every 20th request
        if indexing_progress['phase'] == 'vectors':
            current_count = indexing_progress.get('message_count', 0)
            if current_count % 20 == 0:  # Only update every 20th request
                current_idx = indexing_progress.get('message_index', 0)
                next_idx = (current_idx + 1) % len(VECTOR_MESSAGES)
                indexing_progress['message'] = VECTOR_MESSAGES[next_idx]
                indexing_progress['message_index'] = next_idx
            indexing_progress['message_count'] = current_count + 1

        response = {
            'phase': indexing_progress['phase'],
            'status': indexing_progress['status'],
            'message': indexing_progress['message'],
            'current': indexing_progress['current'],
            'total': indexing_progress['total'],
            'indexed_files': list(indexing_progress['indexed_files'])
        }
        return jsonify(response)
    else:
        return render_template('progress.html', indexing_progress=indexing_progress)

if __name__ == '__main__':
    try:
        if collection_exists():
            logger.info("=== Using existing index ===")
            docsearch = Qdrant(
                client=QdrantClient(url="http://vectorstore:6333"),
                collection_name="code_chunks",
                embeddings=SentenceTransformerEmbeddings()
            )
        else:
            logger.info("=== No index found. Waiting for user to initiate indexing ===")
        
        logger.info("=================================================")
        logger.info("=== Code Search is ready at http://localhost:5000 ===")
        logger.info("=================================================")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        exit(1)

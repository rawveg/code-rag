from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, session
import logging
import os
from src.embeddings import SentenceTransformerEmbeddings, format_code_for_display
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from qdrant_client import QdrantClient
from threading import Thread, Lock
from flask import has_request_context, current_app

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Starting the application...')

repo_path = "/app/repo"

# Global state for indexing progress
indexing_progress = {
    'current': 0,
    'total': 0,
    'status': 'idle',
    'message': '',
    'phase': 'idle',
    'indexed_files': []
}

# Messages to cycle through during vector creation
VECTOR_MESSAGES = [
    'Creating vector index... This may take a few minutes.',
    'Creating vector index... Please wait, still indexing',
    'Creating vector index... Don\'t leave this page',
    'Creating vector index... Your patience is appreciated'
]

# Add locks for thread safety
indexing_lock = Lock()
docsearch_lock = Lock()

# Global variable for the search index
docsearch = None

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'  # Make sure static folder is configured
app.secret_key = os.urandom(24)  # Or use a fixed secret key if you prefer

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
    'venv',
    'storage',
    
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
    '.py', '.js', '.ts', '.php', '.html', '.twig', '.yaml', '.yml', '.json'  # Just file extensions
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

# Default settings
DEFAULT_SETTINGS = {
    'file_patterns': ['.py', '.js', '.ts', '.php', '.html', '.twig', '.yaml', '.yml', '.json'],
    'skip_dirs': list(SKIP_DIRS)
}

# Add debug logging for directory skipping
def should_skip_directory(dirname, skip_dirs=None):
    """Check if a directory should be skipped during indexing."""
    skip_dirs = skip_dirs or DEFAULT_SETTINGS['skip_dirs']
    should_skip = dirname.lower() in skip_dirs or 'cache' in dirname.lower()
    if should_skip:
        logger.debug(f"Skipping directory: {dirname}")
    return should_skip

# Move all the indexing code into a function
def initialize_search(settings=None):
    """Initialize the search index with the given settings."""
    global indexing_progress
    
    # Use provided settings or defaults
    settings = settings or DEFAULT_SETTINGS
    
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
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if not should_skip_directory(d, settings['skip_dirs'])]
            
            # Log files being counted
            valid_files = [f for f in files if f.endswith(tuple(settings['file_patterns']))]
            total_files += len(valid_files)

        logger.info(f"Total files to process: {total_files}")
        
        indexing_progress['total'] = total_files
        indexing_progress['current'] = 0
        indexing_progress['message'] = 'Processing files...'
        indexing_progress['indexed_files'] = []

        texts = []
        metadatas = []
        
        # Process files with progress
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not should_skip_directory(d, settings['skip_dirs'])]
            
            for file in files:
                if file.endswith(tuple(settings['file_patterns'])):
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
                                current_line += chunk_lines

                            texts.extend(chunks)
                            metadatas.extend([{
                                "source": rel_path,
                                "line_start": line_nums['start'],
                                "line_end": line_nums['end']
                            } for line_nums in line_numbers])
                            indexing_progress['indexed_files'].append(rel_path)
                    except Exception as e:
                        logger.error(f"Error reading file {rel_path}: {e}")

        logger.info("File processing complete, starting vector creation...")
        indexing_progress.update({
            'phase': 'vectors',
            'message': VECTOR_MESSAGES[0],
            'message_index': 0,
            'message_count': 0,
            'current': 0,
            'total': len(texts)
        })
        
        # Process texts in batches to show progress
        batch_size = 100
        processed_texts = []
        processed_metadatas = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            # Update progress
            indexing_progress.update({
                'current': i,
                'total': len(texts),
                'message': VECTOR_MESSAGES[indexing_progress.get('message_index', 0)]
            })
            logger.info(f"Processing vectors batch: {i}/{len(texts)}")
            
            # Process batch
            processed_texts.extend(batch_texts)
            processed_metadatas.extend(batch_metadatas)
        
        # Create the final vector store
        result = Qdrant.from_texts(
            processed_texts,
            embeddings,
            metadatas=processed_metadatas,
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

def background_reindex():
    """Run reindexing in a background thread."""
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
        
        # Clear existing index first
        response = requests.delete("http://vectorstore:6333/collections/code_chunks")
        if response.status_code != 200:
            raise Exception(f"Failed to clear index: {response.status_code}")
        
        # Initialize search with thread safety
        with docsearch_lock:
            docsearch = initialize_search(DEFAULT_SETTINGS)
        
        # Mark as complete
        indexing_progress.update({
            'phase': 'complete',
            'status': 'complete',
            'message': 'Indexing complete'
        })
        
    except Exception as e:
        logger.error(f"Error during reindex: {e}")
        indexing_progress.update({
            'status': 'error',
            'phase': 'error',
            'message': str(e)
        })

@app.route('/admin/reindex', methods=['POST'])
def force_reindex():
    """Force a reindex of the codebase."""
    try:
        # Start background thread
        thread = Thread(target=background_reindex)
        thread.daemon = True
        thread.start()
        return redirect(url_for('show_progress'))
    except Exception as e:
        logger.error(f"Error starting reindex: {e}")
        flash(f"Error starting reindex: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/admin/progress')
def show_progress():
    """Show the current indexing progress."""
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if is_ajax:
        # Rotate message if in vector phase
        if indexing_progress['phase'] == 'vectors':
            current_count = indexing_progress.get('message_count', 0)
            if current_count % 20 == 0:  # Update message every 20th request
                current_idx = indexing_progress.get('message_index', 0)
                next_idx = (current_idx + 1) % len(VECTOR_MESSAGES)
                indexing_progress['message'] = VECTOR_MESSAGES[next_idx]
                indexing_progress['message_index'] = next_idx
            indexing_progress['message_count'] = current_count + 1
            
        return jsonify({
            'phase': indexing_progress['phase'],
            'status': indexing_progress['status'],
            'message': indexing_progress['message'],
            'current': indexing_progress['current'],
            'total': indexing_progress['total'],
            'indexed_files': list(indexing_progress['indexed_files'])
        })
    
    return render_template('progress.html', indexing_progress=indexing_progress)

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
        # Remove the forced JWT context
        docs = docsearch.similarity_search_with_score(query, k=10)
        logger.info(f"Found {len(docs)} initial results")
        
        # Prepare results with more relaxed filtering
        results = []
        seen = set()
        for doc, score in docs:
            if score < 0.5:  # Adjust threshold to be more lenient
                content = doc.page_content.strip()
                if content not in seen:
                    seen.add(content)
                    formatted_content = format_code_for_display(
                        content, 
                        'php',  # You might want to detect language from file extension
                        line_start=doc.metadata.get('line_start', 1)
                    )
                    results.append({
                        'filepath': doc.metadata.get('source', 'unknown'),
                        'content': formatted_content,
                        'raw_content': content,  # Store the unformatted code
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
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        return False

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

@app.route('/admin/settings')
def show_settings():
    # Get current settings or use defaults
    settings = session.get('settings', DEFAULT_SETTINGS)
    return render_template('settings.html', settings=settings)

@app.route('/admin/settings/save', methods=['POST'])
def save_settings():
    settings = {
        'skip_dirs': request.form.get('skip_dirs', '').splitlines(),
        'file_patterns': request.form.get('file_patterns', '').splitlines(),
        'priority_paths': request.form.get('priority_paths', '').splitlines()
    }
    
    # Clean up empty lines
    for key in settings:
        settings[key] = [x.strip() for x in settings[key] if x.strip()]
    
    session['settings'] = settings
    flash('Settings saved successfully', 'success')
    return redirect(url_for('show_settings'))

@app.route('/admin/settings/reset', methods=['POST'])
def reset_settings():
    session['settings'] = DEFAULT_SETTINGS.copy()
    return jsonify({'success': True})

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

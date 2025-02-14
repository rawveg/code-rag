# Code RAG

A semantic code search tool powered by RAG (Retrieval Augmented Generation) that helps you find relevant code in your codebase by asking questions in natural language.

## Features

- 🔍 Natural language code search
- 📝 Semantic understanding of code
- 🚀 Fast vector-based retrieval
- 💡 Context-aware results
- 📊 Progress tracking for indexing
- 🎨 Syntax highlighting for results

## Tech Stack

- **Backend**: Flask (Python)
- **Vector Store**: Qdrant
- **Embeddings**: Sentence Transformers
- **Frontend**: HTML/CSS/JavaScript
- **Code Processing**: Pygments for syntax highlighting

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Git (for cloning repositories to search)

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/rawveg/code-rag.git
   cd code-rag
   ```

2. Configure repository path:
   - Edit `docker-compose.yaml` to map your repository directory:
     ```yaml
     volumes:
       - /path/to/repository:/app/repo
     ```

3. Build and start the containers:
   ```bash
   docker-compose up --build -d
   ```

4. Visit `http://localhost:5000` in your browser

5. Click "Index Repository" to start indexing your codebase

## Usage

1. **Index Your Code**:
   - First-time setup requires manual indexing
   - Click "Index Repository" to start the initial indexing process
   - Progress is shown in real-time as files are processed
   - Vector embeddings are persisted between restarts

2. **Search Your Code**:
   - Type natural language questions about your code
   - Example: "How do we communicate with S3 buckets?"
   - Click Submit to see relevant code snippets

3. **View Results**:
   - Results show file paths and relevant code sections
   - Syntax highlighting helps readability
   - Copy paths or code snippets with one click

## Configuration

The tool can be configured through the settings page (⚙️ icon) with the following options:

### Directory Exclusions
Directories that will be skipped during indexing. By default, this includes:
- Package directories (node_modules, vendor, etc.)
- Build outputs and caches (dist, build, __pycache__)
- Version control (.git, .svn)
- Documentation folders (docs, doc)
- Test directories (tests, test)
- IDE files (.idea, .vscode)
- Asset directories (images, fonts, etc.)

### File Patterns
File extensions to include in the search index. Default patterns include:
- `.py` (Python files)
- `.js` (JavaScript files)
- `.ts` (TypeScript files)
- `.php` (PHP files)
- `.html` (HTML files)
- `.twig` (Twig templates)
- `.yaml`, `.yml` (YAML files)
- `.json` (JSON files)

### Priority Paths
Directories to prioritize during search. These paths are given higher relevance in search results.

All settings can be modified through the UI and reset to defaults if needed. Changes require re-indexing to take effect.

## Indexing Behavior

The application uses a "lazy indexing" approach for better performance and user experience:

- **First Run**: No automatic indexing on startup
  - This ensures fast application startup
  - Gives you control over when to start the resource-intensive indexing process

- **Subsequent Runs**: Uses existing index
  - Vector embeddings persist between restarts
  - No need to reindex unless code has changed
  - Faster startup by reusing existing index

- **Manual Control**:
  - "Index Repository" - Start initial indexing or reindex
  - "Force Reindex" - Clear and rebuild the index
  - "Clear Index" - Remove all indexed data

## Development

To run the project in development mode:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask development server:
   ```bash
   python app.py
   ```

## Roadmap

### Smart Indexing
- **Partial Reindexing**
  - Track file hashes to detect changes
  - Only reindex modified files
  - Handle file renames and moves
- **Git Integration**
  - Branch-aware indexing
  - Track indexed state per branch
  - Use git diff for smart updates
  - Direct repository connection
    - Clone repositories directly from Git
    - Support for GitHub/GitLab/Bitbucket
    - Authentication for private repositories

### Performance Improvements
- **Parallel Processing**
  - Multi-threaded file processing
  - Batch vector creation
- **Caching Layer**
  - Cache frequent queries
  - Store preprocessed results

### UI Enhancements
- **Advanced Search Options**
  - Filter by file types
  - Exclude specific paths
  - Sort by relevance/date

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
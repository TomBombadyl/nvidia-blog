# NVIDIA Blog MCP Server

A production-ready [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that provides searchable access to NVIDIA's official developer and blog content. This server enables AI assistants like Cursor to search and retrieve information from NVIDIA's extensive blog archives with grounded, factual responses.

## Features

- 🔍 **Dual Search Methods**: RAG (Retrieval-Augmented Generation) and Vector Search
- 🧠 **AI-Powered Query Enhancement**: Automatic query transformation and answer grading
- 📚 **Comprehensive Coverage**: Access to 100+ NVIDIA blog posts from developer.nvidia.com and blogs.nvidia.com
- 🔄 **Always Up-to-Date**: Daily automated ingestion keeps the database current with latest NVIDIA content
- ✅ **Production Ready**: Deployed on Google Cloud Run with high availability
- 🎯 **Grounded Responses**: All answers include source citations from official NVIDIA blogs

## Quick Start for Cursor

Add this to your Cursor MCP configuration file (usually `~/.cursor/mcp.json` or `%APPDATA%\Cursor\User\mcp.json`):

```json
{
  "mcpServers": {
    "nvidia-blog": {
      "url": "https://nvidia-blog-mcp-server-4vvir4xvda-ey.a.run.app/mcp",
      "transport": "streamable-http"
    }
  }
}
```

After adding this configuration, restart Cursor. You can then ask questions like:
- "What's new in robotics from NVIDIA?"
- "How do I optimize CUDA kernels?"
- "What are the latest TensorRT features?"

## How It Works

### Data Pipeline

The server maintains an up-to-date database through a daily automated ingestion pipeline:

1. **RSS Feed Collection**: Fetches new posts from NVIDIA's official RSS feeds
2. **Content Processing**: Cleans and processes HTML content into searchable text
3. **AI Indexing**: Ingests content into Vertex AI RAG Corpus and Vector Search
4. **Daily Updates**: Runs automatically every day at 7:00 AM UTC

### Search Capabilities

**RAG Method** (default):
- Returns full text chunks with source citations
- Includes query transformation for better results
- Answer grading ensures quality responses
- Best for comprehensive answers with citations

**Vector Search Method**:
- Semantic similarity search
- Fast retrieval of related content
- Returns document IDs with similarity scores
- Best for finding conceptually similar content

## Usage Examples

### Basic Query
```
"What are CUDA programming best practices?"
```

### Specific Technology
```
"Tell me about TensorRT inference optimization"
```

### Latest Updates
```
"What's new in autonomous driving from NVIDIA?"
```

### Research Topics
```
"How does NVIDIA approach multi-GPU training?"
```

## Project Structure

```
nvidia_blog/
├── mcp/                      # MCP server implementation
│   ├── mcp_server.py         # Main MCP server
│   ├── mcp_service.py        # Cloud Run service entry point
│   ├── query_rag.py          # RAG Corpus query module
│   ├── query_vector_search.py # Vector Search query module
│   ├── rag_query_transformer.py # Query enhancement
│   ├── rag_answer_grader.py  # Answer quality evaluation
│   └── config.py             # Configuration management
├── Dockerfile.mcp            # Container definition
├── cloudbuild.mcp.yaml       # CI/CD configuration
├── requirements.txt           # Python dependencies
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guidelines
└── SECURITY.md               # Security policy
```

## Configuration

The server uses environment variables for configuration. See `.env.example` for all available options.

Key configuration variables:
- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `GCP_REGION`: GCP region (default: europe-west3)
- `RAG_CORPUS`: Vertex AI RAG Corpus resource path
- `VECTOR_SEARCH_ENDPOINT_ID`: Vector Search endpoint ID
- `VECTOR_SEARCH_INDEX_ID`: Vector Search index ID
- `RAG_VECTOR_DISTANCE_THRESHOLD`: Similarity threshold (default: 0.5)
- `GEMINI_MODEL_LOCATION`: Location for Gemini models (default: europe-west4)

## Development

### Prerequisites

- Python 3.9+
- Google Cloud SDK
- GCP project with Vertex AI API enabled
- Service account with appropriate permissions

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/TomBombadyl/nvidia-blog.git
   cd nvidia-blog
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your GCP configuration
   ```

5. Authenticate with GCP:
   ```bash
   gcloud auth application-default login
   ```

### Running Locally

The MCP server is designed to run on Cloud Run, but you can test the server locally:

```bash
cd mcp
python mcp_service.py
```

**Note**: The server expects all MCP modules to be in the same directory. When running locally, ensure you're in the `mcp/` directory or adjust Python path accordingly.

## Deployment

The server is deployed to Google Cloud Run using Cloud Build. See `cloudbuild.mcp.yaml` for deployment configuration.

```bash
gcloud builds submit --config cloudbuild.mcp.yaml --project=your-project-id
```

## API Response Format

### RAG Response
```json
{
  "query": "original query",
  "transformed_query": "enhanced query",
  "contexts": [
    {
      "text": "retrieved content",
      "source_uri": "https://developer.nvidia.com/blog/...",
      "distance": 0.45
    }
  ],
  "count": 10,
  "grade": {
    "score": 0.85,
    "relevance": 0.90,
    "completeness": 0.80,
    "grounded": true
  }
}
```

## Content Disclaimer

**Important**: This service provides search access to content from NVIDIA Corporation's official blogs. All blog content is the intellectual property of NVIDIA Corporation. This software does not claim ownership of any NVIDIA content. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

If you discover a security vulnerability, please see [SECURITY.md](SECURITY.md) for reporting instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The code is MIT licensed, but all content from NVIDIA blogs remains the property of NVIDIA Corporation. See the LICENSE file for the full third-party content disclaimer.

## Resources

- [MCP Protocol Documentation](https://modelcontextprotocol.io)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [NVIDIA Official Blog](https://blogs.nvidia.com)
- [GitHub Repository](https://github.com/TomBombadyl/nvidia-blog)

## Status

✅ **Operational** - Server is live and serving queries  
📊 **Database**: 100+ blog posts indexed and searchable  
🔄 **Updates**: Daily automated ingestion active  
⚙️ **Search Metrics**: Vector distance threshold 0.5, default 10 neighbors

## Architecture

The MCP server uses a production-ready architecture:

- **Search Methods**: Dual RAG and Vector Search with configurable thresholds
- **Query Enhancement**: Automatic query transformation using Gemini 2.0 Flash
- **Quality Assurance**: Answer grading with iterative refinement (up to 2 iterations)
- **Deployment**: Google Cloud Run with automatic scaling
- **Data Pipeline**: Daily automated RSS ingestion keeps content current

## Performance

- **Default Results**: 10 neighbors/contexts per query
- **Distance Threshold**: 0.5 (balanced for recall and precision)
- **Response Time**: Sub-second for most queries
- **Availability**: High availability on Cloud Run
- **Gemini Integration**: Query transformation and grading via europe-west4 (Netherlands, closest to RAG corpus in europe-west3 for data residency)

---

Built for developers by developers
"Let's go find your wand" - TB

<img width="2752" height="1536" alt="NVIDIA BLOG MCP Thumbnail" src="https://github.com/user-attachments/assets/99200cd5-423d-4178-81b1-146170996618" />


# NVIDIA Blog MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that provides AI assistants like Cursor with grounded access to NVIDIA's official developer and blog content through Retrieval-Augmented Generation (RAG).

## Features

- **Grounded Context with RAG**: Retrieval-Augmented Generation provides responses based on content from official NVIDIA blogs
- **Multilingual Support**: Uses `text-multilingual-embedding-002` for semantic search across 50+ languages
- **AI-Powered Query Enhancement**: Automatic query transformation and answer grading for higher quality results
- **Current Content**: Blog posts indexed from December 1, 2025 onwards. Older content is not yet ingested; all future posts will be automatically added via daily updates
- **Accurate Responses**: Built-in answer quality evaluation prevents hallucinations

## Quick Start for Cursor

In Cursor settings, add this .json block to your Cursor MCP configuration (usually `~/.cursor/mcp.json` or `%APPDATA%\Cursor\User\mcp.json`):

```json
{
  "mcpServers": {
    "nvidia-blog": {
      "url": "https://nvidia-blog-mcp-xxx.run.app/mcp",
      "transport": "streamable-http"
    }
  }
}
```

**Note**: The service URL will be updated after deployment. The previous RAG Corpus-based MCP has been archived in favor of the BigQuery-based implementation for improved performance and date-aware searching.

After adding this configuration, restart Cursor. You can then ask questions about NVIDIA technologies, and the server will search the NVIDIA blog archives to provide grounded answers based on official NVIDIA content.

## How It Works

### Data Pipeline

The server maintains a database through a daily automated ingestion pipeline:

1. **RSS Feed Collection**: Fetches new posts from NVIDIA's official RSS feeds
2. **Content Processing**: Cleans and processes HTML content into searchable text
3. **BigQuery Indexing**: Ingests content into BigQuery with vector embeddings for semantic search
4. **Daily Updates**: Runs automatically every day at 7:00 AM UTC

**Note**: The previous RAG Corpus-based implementation has been archived. The current implementation uses BigQuery with ML.DISTANCE for improved performance and date-aware filtering capabilities.

### Response Quality

All responses are graded for:
- **Relevance**: How well the retrieved content matches the query
- **Completeness**: Whether the answer covers the topic adequately
- **Grounding**: All answers are based on content from official NVIDIA blog posts

## Project Structure

```
nvidia_blog/
├── bigquery/
│   └── mcp_server/           # Primary MCP server implementation (BigQuery-based)
│       ├── mcp_server.py      # Main MCP server
│       ├── mcp_service.py     # Cloud Run service entry point
│       ├── query_bigquery.py  # BigQuery query module with ML.DISTANCE
│       ├── date_filter_extractor.py # Date-aware query filtering
│       └── config.py          # Configuration management
├── archive/
│   └── mcp-rag-corpus/       # Archived RAG Corpus implementation
│       ├── mcp_server.py      # Original RAG Corpus server
│       ├── query_rag.py       # RAG Corpus query module
│       ├── rag_query_transformer.py # Query enhancement
│       └── rag_answer_grader.py # Answer quality evaluation
├── Dockerfile.mcp            # Container definition
├── cloudbuild.mcp.yaml       # CI/CD configuration
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── NOTICE                    # Third-party content notice
├── CONTRIBUTING.md           # Contribution guidelines
└── SECURITY.md               # Security policy
```

## Content Disclaimer

**Important**: This service provides search access to content from NVIDIA Corporation's official blogs. All blog content is the intellectual property of NVIDIA Corporation. This software does not claim ownership of any NVIDIA content. 

**SynapGarden is not affiliated with, endorsed by, or sponsored by NVIDIA Corporation.** This is a free and open-source project developed independently by SynapGarden.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full details.

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
- [GitHub Repository](https://github.com/SynapGarden/NVIDIA_blog_mcp)

## Acknowledgments

SynapGarden is a member of the NVIDIA Inception program and Google Cloud for Startups program. We thank them for providing resources that enable us to build useful tools for the developer community.

**Note**: Program membership does not imply endorsement, sponsorship, or affiliation with NVIDIA Corporation or Google Cloud beyond program participation.

---

Developed by [SynapGarden](https://github.com/SynapGarden) | Free and Open Source

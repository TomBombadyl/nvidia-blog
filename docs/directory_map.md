NVIDIA Blog MCP - Complete Codebase Directory Map
==================================================

z:\SynapGarden\nvidia_blog/
│
├── 📄 README.md                          # Project overview and setup instructions
├── 📄 LICENSE                            # MIT License
├── 📄 NOTICE                             # Third-party content notice
├── 📄 SECURITY.md                        # Security policy and reporting
├── 📄 CONTRIBUTING.md                    # Contribution guidelines
│
├── 📄 requirements.txt                   # Python dependencies (feedparser, requests, beautifulsoup4, google-cloud-*, vertexai, mcp, pydantic, uvicorn, starlette)
├── 📄 Dockerfile.mcp                     # Container for MCP Server Cloud Run service
├── 📄 cloudbuild.mcp.yaml                # CI/CD config for MCP Server deployment
├── 📄 CREATE_RAG_INDEX_STEPS.md          # Step-by-step guide for RAG Corpus setup
│
├── 📁 mcp/                               # MCP Server Implementation (Read-only Query Interface)
│   ├── 📄 config.py                      # Configuration management (RAG_VECTOR_DISTANCE_THRESHOLD=0.7, Gemini, RSS feeds)
│   ├── 📄 mcp_server.py                  # Main MCP server implementation (search_nvidia_blogs tool)
│   ├── 📄 mcp_service.py                 # Cloud Run service entry point (uvicorn startup)
│   ├── 📄 query_rag.py                   # **CRITICAL** RAG Corpus query module with transformation & grading
│   ├── 📄 query_vector_search.py         # Vector Search query module (semantic similarity)
│   ├── 📄 rag_query_transformer.py       # Query enhancement with date awareness
│   └── 📄 rag_answer_grader.py           # Answer quality evaluation
│
├── 📁 private/                           # RSS Ingestion Pipeline (Write Operations - Daily Scheduled Job)
│   ├── 📄 main.py                        # Cloud Run Job entry point (orchestrates ingestion)
│   ├── 📄 rss_fetcher.py                 # RSS feed fetching and parsing
│   ├── 📄 html_cleaner.py                # HTML cleaning with date metadata embedding
│   ├── 📄 gcs_utils.py                   # Google Cloud Storage utilities (read/write JSON and files)
│   ├── 📄 rag_ingest.py                  # RAG Corpus ingestion via REST API (chunk_size=768, overlap=128)
│   ├── 📄 vector_search_ingest.py        # Vector embedding and upsert to Vector Search index
│   ├── 📄 Dockerfile                     # Container for ingestion Cloud Run Job
│   └── 📄 cloudbuild.yaml                # CI/CD config for ingestion job deployment
│
└── 📁 assets/                            # Screenshot/image assets (Cursor workspace images)
    └── [14 image files]

===========================================

KEY COMPONENTS SUMMARY:

1. MCP SERVER (Cloud Run Service - Read Only)
   - Handles user queries
   - Uses RAG Corpus for retrieval
   - Applies query transformation
   - Grades answer quality
   - Default threshold: 0.7 (configurable)

2. INGESTION PIPELINE (Cloud Run Job - Scheduled Daily)
   - Fetches RSS feeds from NVIDIA blogs
   - Cleans HTML content
   - Embeds publication dates in text
   - Ingests to RAG Corpus (Vertex AI)
   - Upserts vectors to Vector Search
   - Tracks processed items to avoid duplicates

3. CONFIGURATION
   - RAG_VECTOR_DISTANCE_THRESHOLD: 0.7 (from config.py)
   - Region: europe-west3
   - Gemini Location: europe-west4
   - RSS Feeds: developer.nvidia.com/blog + blogs.nvidia.com

4. CRITICAL FIX (Dec 11, 2025)
   - query_rag.py now imports RAG_VECTOR_DISTANCE_THRESHOLD from config
   - Default parameters use config value (0.7) not hardcoded (0.5)
   - Ensures consistent threshold across all query paths

===========================================
# Public Repository Cleanup Plan

**Date**: December 2025  
**Goal**: Prepare NVIDIA Blog MCP Server repository for public release

## Current Status Assessment

### ✅ What's Good
- Code uses environment variables (no hardcoded secrets)
- Well-structured codebase with clear separation of concerns
- Comprehensive technical documentation
- `.gitignore` properly configured for sensitive files
- Test files present for validation

### ⚠️ What Needs Cleanup

#### Files to Remove
1. **Assets folder** - Contains temporary Cursor IDE images:
   - `assets/c__Users_tobin_AppData_Roaming_Cursor_User_workspaceStorage_.../image-*.png`
   - These are temporary IDE files, not needed in repo

2. **Deleted files** (from git status):
   - `docs/DEPLOYMENT_CHECKLIST.md` (deleted)
   - `docs/PROJECT_OVERVIEW.md` (deleted)
   - `docs/SETUP_WINDOWS.md` (deleted)
   - `scripts/scheduler.sh` (deleted)
   - `scripts/setup_gcp_resources.ps1` (deleted)
   - `scripts/setup_gcp_resources.sh` (deleted)
   - `tests/__init__.py` (deleted)
   - `tests/test_local.py` (deleted)
   - Need to commit these deletions

#### Files to Review/Update
1. **README.md** - Currently focused on ingestion pipeline, needs:
   - Clear project description for public audience
   - MCP server usage instructions
   - Public setup instructions (without internal GCP details)
   - Links to documentation

2. **docs/MCP_SERVER_TECHNICAL_REPORT.md** - Contains:
   - Internal GCP resource IDs (should be examples or removed)
   - Internal project details (can keep but mark as examples)

3. **config.py** - Good! Uses env vars, but default project ID should be documented as example

4. **Test files** - `test_mcp_queries.py` is useful, keep it

#### Files to Create
1. **LICENSE** - Need to choose license (MIT, Apache 2.0, etc.)
2. **CONTRIBUTING.md** - Contribution guidelines
3. **.env.example** - Template for environment variables
4. **SECURITY.md** - Security policy (GitHub best practice)
5. **CODE_OF_CONDUCT.md** - Optional but recommended for public repos

## Cleanup Steps

### Phase 1: Commit Current Working State
```bash
# Stage all current changes
git add -A

# Commit with descriptive message
git commit -m "feat: Working MCP server with RAG and Vector Search

- Fully operational MCP server deployed to Cloud Run
- RAG Corpus with 112 blog posts indexed
- Vector Search with 695 vectors indexed
- Query transformation and answer grading
- Daily automated RSS ingestion pipeline
- Comprehensive technical documentation"

# Push to GitHub
git push origin main
```

### Phase 2: Remove Unnecessary Files
```bash
# Remove assets folder (temporary IDE images)
git rm -r assets/

# Commit deletions
git commit -m "chore: Remove temporary assets and deleted files"
```

### Phase 3: Add Public Repo Essentials

#### 3.1 Create LICENSE File
**Question for user**: What license? Common choices:
- MIT (permissive, most popular)
- Apache 2.0 (permissive with patent grant)
- GPL v3 (copyleft)

#### 3.2 Create .env.example
Template showing required environment variables without values.

#### 3.3 Create CONTRIBUTING.md
Guidelines for:
- How to contribute
- Code style
- Testing requirements
- Pull request process

#### 3.4 Create SECURITY.md
Security policy for reporting vulnerabilities.

### Phase 4: Update Documentation

#### 4.1 Update README.md
- Add clear project description
- Add MCP server usage section
- Add public setup instructions
- Remove or generalize internal GCP details
- Add badges (optional)
- Add links to docs

#### 4.2 Update MCP_SERVER_TECHNICAL_REPORT.md
- Mark internal resource IDs as examples
- Add note that these are example values
- Keep technical details but make them generic

### Phase 5: Final Verification
- [ ] No hardcoded secrets
- [ ] No personal/internal information
- [ ] All sensitive files in .gitignore
- [ ] LICENSE file present
- [ ] README is public-friendly
- [ ] Documentation is clear
- [ ] Test files work (if keeping)

## Best Practices Applied

Based on GitHub documentation:
- ✅ README in root directory
- ✅ LICENSE file in root
- ✅ .gitignore properly configured
- ✅ Environment variables for configuration
- ⏳ CONTRIBUTING.md (to be created)
- ⏳ SECURITY.md (to be created)
- ⏳ Clear project description
- ⏳ Public setup instructions

## Files Structure After Cleanup

```
nvidia_blog/
├── .dockerignore
├── .gitignore
├── .env.example                    # NEW
├── LICENSE                         # NEW
├── CONTRIBUTING.md                 # NEW
├── SECURITY.md                     # NEW
├── README.md                       # UPDATED
├── requirements.txt
├── config.py
├── cloudbuild.mcp.yaml
├── cloudbuild.yaml
├── Dockerfile
├── Dockerfile.mcp
├── main.py
├── mcp_server.py
├── mcp_service.py
├── query_rag.py
├── query_vector_search.py
├── rag_answer_grader.py
├── rag_ingest.py
├── rag_query_transformer.py
├── rss_fetcher.py
├── html_cleaner.py
├── gcs_utils.py
├── vector_search_ingest.py
├── test_mcp_queries.py
├── docs/
│   └── MCP_SERVER_TECHNICAL_REPORT.md  # UPDATED
└── (assets/ removed)
```

## Next Steps

1. **User Decision**: Choose license type
2. **Execute Phase 1**: Commit and push current state
3. **Execute Phase 2**: Remove unnecessary files
4. **Execute Phase 3**: Create public repo files
5. **Execute Phase 4**: Update documentation
6. **Execute Phase 5**: Final verification
7. **Make Public**: Change repo visibility on GitHub

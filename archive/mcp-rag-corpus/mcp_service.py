"""
MCP Server Entry Point for Cloud Run Service
Uses FastMCP's streamable HTTP app with uvicorn for Cloud Run compatibility.
For stateless HTTP servers, no session manager is required.
"""

import os
import sys
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse

# Get port from environment (Cloud Run sets PORT automatically, default to 8080)
PORT = int(os.environ.get("PORT", 8080))

# Print startup info immediately - this must work or container fails
print("=" * 50, flush=True)
print("NVIDIA Blog MCP Server Starting...", flush=True)
print(f"Port: {PORT}", flush=True)
print("Python version:", sys.version, flush=True)
print("Working directory:", os.getcwd(), flush=True)
print("Python path:", sys.path, flush=True)
print("=" * 50, flush=True)

# Verify critical imports work
try:
    print("Verifying imports...", flush=True)
    import starlette
    import uvicorn
    print(f"✅ starlette: {starlette.__version__ if hasattr(starlette, '__version__') else 'ok'}", flush=True)
    print(f"✅ uvicorn: {uvicorn.__version__}", flush=True)
except Exception as e:
    print(f"❌ Import verification failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Root health endpoint handler
async def root_health(request):
    """Root health check endpoint."""
    return JSONResponse({"status": "ok", "service": "nvidia-blog-mcp-server"})

if __name__ == "__main__":
    try:
        print("Step 1: Importing MCP server...", flush=True)
        from mcp_server import mcp
        print("✅ MCP server imported", flush=True)
        
        print("Step 2: Creating streamable HTTP app...", flush=True)
        try:
            print(f"   streamable_http_path setting: {mcp.settings.streamable_http_path} (default: /mcp)", flush=True)
            app = mcp.streamable_http_app()
            print("✅ Streamable HTTP app created", flush=True)
            print(f"   app type: {type(app)}", flush=True)
        except Exception as e:
            print(f"❌ ERROR in Step 2: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # For stateless HTTP, no session manager lifespan is needed
        # The streamable_http_app() already returns a fully functional Starlette app
        
        print("Step 3: Configuring app routes...", flush=True)
        try:
            # Add explicit root route for health checks
            app.add_route("/", root_health, methods=["GET"])
            print("✅ Root route added", flush=True)
            
            # Add request logging middleware to debug incoming requests
            @app.middleware("http")
            async def log_requests(request, call_next):
                """Log all incoming requests for debugging."""
                import time
                start_time = time.time()
                response = await call_next(request)
                process_time = time.time() - start_time
                print(f"   Request: {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)", flush=True)
                return response
            
            print("✅ Request logging middleware added", flush=True)
            
            # Debug: Inspect routes via router
            print("   Debug: Inspecting routes...", flush=True)
            try:
                if hasattr(app, 'router') and hasattr(app.router, 'routes'):
                    routes = app.router.routes
                    print(f"     Router has {len(routes)} routes", flush=True)
                    for j, route in enumerate(routes[:15]):
                        route_info = f"       Route {j}: {type(route).__name__}"
                        if hasattr(route, 'path'):
                            route_info += f" path='{route.path}'"
                        elif hasattr(route, 'path_regex'):
                            route_info += f" regex='{route.path_regex.pattern}'"
                        if hasattr(route, 'methods'):
                            route_info += f" methods={route.methods}"
                        print(route_info, flush=True)
                else:
                    print("     WARNING: Could not find routes via router", flush=True)
            except Exception as route_e:
                print(f"     Could not inspect routes: {route_e}", flush=True)
                import traceback
                traceback.print_exc()
            
            print("✅ App configured", flush=True)
            print(f"   streamable_http_path: {mcp.settings.streamable_http_path}", flush=True)
            print(f"   MCP endpoint: POST {mcp.settings.streamable_http_path}", flush=True)
            print(f"   Health endpoint: GET /health", flush=True)
            print(f"   Root endpoint: GET /", flush=True)
        except Exception as e:
            print(f"❌ ERROR in Step 3: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
    except Exception as e:
        print(f"❌ FATAL: Failed to initialize MCP: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Fallback: minimal health check app
        app = Starlette(routes=[])
        app.add_route("/", root_health, methods=["GET"])
        print("⚠️  Using minimal health check app only", flush=True)
    
    try:
        print("Step 4: Starting uvicorn server...", flush=True)
        print(f"Port: {PORT}", flush=True)
        print(f"Endpoints:", flush=True)
        print(f"  - GET  / (root health)", flush=True)
        print(f"  - GET  /health (MCP health)", flush=True)
        print(f"  - POST /mcp (MCP protocol)", flush=True)
        print("=" * 50, flush=True)
        print("Server starting - listening for requests...", flush=True)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("Server stopped by user", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"❌ FATAL: Failed to start uvicorn: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

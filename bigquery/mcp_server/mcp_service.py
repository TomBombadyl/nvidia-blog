"""
MCP Server Entry Point for Cloud Run Service (BigQuery)
Uses FastMCP's streamable HTTP app with uvicorn for Cloud Run compatibility.
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
    return JSONResponse({"status": "ok", "service": "nvidia-blog-mcp"})

if __name__ == "__main__":
    try:
        print("Step 1: Importing MCP server...", flush=True)
        # #region agent log
        import json as json_lib
        log_data = {"location": "mcp_service.py:44", "message": "Starting MCP server import", "data": {"project_id": os.getenv("GCP_PROJECT_ID"), "region": os.getenv("GCP_REGION"), "python_path": sys.path}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "runId": "startup-debug", "hypothesisId": "C"}
        print(f"[DEBUG] {json_lib.dumps(log_data)}", flush=True)
        # #endregion
        # Add parent directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from bigquery.mcp_server.mcp_server import mcp
        print("✅ MCP server imported", flush=True)
        # #region agent log
        log_data = {"location": "mcp_service.py:50", "message": "MCP server imported successfully", "data": {"streamable_http_path": mcp.settings.streamable_http_path}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "runId": "startup-debug", "hypothesisId": "C"}
        print(f"[DEBUG] {json_lib.dumps(log_data)}", flush=True)
        # #endregion
        
        print("Step 2: Creating streamable HTTP app...", flush=True)
        try:
            print(f"   streamable_http_path setting: {mcp.settings.streamable_http_path} (default: /mcp)", flush=True)
            # #region agent log
            log_data = {"location": "mcp_service.py:55", "message": "Creating streamable HTTP app", "data": {"streamable_http_path": mcp.settings.streamable_http_path}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "runId": "startup-debug", "hypothesisId": "D"}
            print(f"[DEBUG] {json_lib.dumps(log_data)}", flush=True)
            # #endregion
            app = mcp.streamable_http_app()
            print("✅ Streamable HTTP app created", flush=True)
            print(f"   app type: {type(app)}", flush=True)
            # #region agent log
            log_data = {"location": "mcp_service.py:60", "message": "Streamable HTTP app created", "data": {"app_type": str(type(app)), "has_routes": hasattr(app, "routes")}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "runId": "startup-debug", "hypothesisId": "D"}
            print(f"[DEBUG] {json_lib.dumps(log_data)}", flush=True)
            # #endregion
        except Exception as e:
            print(f"❌ ERROR in Step 2: {e}", flush=True)
            import traceback
            # #region agent log
            log_data = {"location": "mcp_service.py:65", "message": "Error creating streamable HTTP app", "data": {"error": str(e), "traceback": traceback.format_exc()}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "runId": "startup-debug", "hypothesisId": "E"}
            print(f"[DEBUG] {json_lib.dumps(log_data)}", flush=True)
            # #endregion
            traceback.print_exc()
            raise
        
        print("Step 3: Configuring app routes...", flush=True)
        try:
            import json
            import time
            import os
            # Use /tmp for Cloud Run (Linux) or workspace path for local
            if os.path.exists("/tmp"):
                log_path = "/tmp/debug.log"
            else:
                # Local development
                log_path = os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", ".cursor", "debug.log"
                )
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # #region agent log
            log_data = {
                "location": "mcp_service.py:63",
                "message": "Inspecting app routes before modification",
                "data": {
                    "app_type": str(type(app)),
                    "has_routes": hasattr(app, "routes")
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "route-debug",
                "hypothesisId": "A"
            }
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass  # Fail silently if log file can't be written
            # #endregion

            # Inspect existing routes
            if hasattr(app, "routes"):
                print(f"   Existing routes count: {len(app.routes)}",
                      flush=True)
                for i, route in enumerate(app.routes):
                    route_info = {
                        "path": getattr(route, "path", "unknown"),
                        "methods": getattr(route, "methods", [])
                    }
                    print(f"   Route {i}: {route_info}", flush=True)
                    # #region agent log
                    log_data = {
                        "location": "mcp_service.py:70",
                        "message": "Found existing route",
                        "data": route_info,
                        "timestamp": int(time.time() * 1000),
                        "sessionId": "debug-session",
                        "runId": "route-debug",
                        "hypothesisId": "A"
                    }
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(log_data) + "\n")
                    except Exception:
                        pass
                    # #endregion
            else:
                print("   ⚠️  App has no 'routes' attribute",
                      flush=True)
                # #region agent log
                log_data = {
                    "location": "mcp_service.py:77",
                    "message": "App missing routes attribute",
                    "data": {"app_attrs": dir(app)},
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "route-debug",
                    "hypothesisId": "B"
                }
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data) + "\n")
                except Exception:
                    pass
                # #endregion

            # Add explicit root route for health checks
            app.add_route("/", root_health, methods=["GET"])
            print("✅ Root route added", flush=True)

            # Manually add health route
            async def health_handler(request):
                """Health check endpoint handler."""
                # #region agent log
                log_data = {
                    "location": "mcp_service.py:87",
                    "message": "Health endpoint called",
                    "data": {
                        "path": str(request.url.path),
                        "method": request.method
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "route-debug",
                    "hypothesisId": "C"
                }
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data) + "\n")
                except Exception:
                    pass
                # #endregion
                return JSONResponse({
                    "status": "healthy",
                    "server": "NVIDIA Developer Resources Search"
                })

            app.add_route("/health", health_handler, methods=["GET"])
            print("✅ Health route manually added", flush=True)

            # Add request logging middleware
            @app.middleware("http")
            async def log_requests(request, call_next):
                """Log all incoming requests for debugging."""
                start_time = time.time()
                # #region agent log
                log_data = {
                    "location": "mcp_service.py:100",
                    "message": "Incoming request",
                    "data": {
                        "method": request.method,
                        "path": str(request.url.path),
                        "query": str(request.url.query),
                        "headers": dict(request.headers)
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "route-debug",
                    "hypothesisId": "D"
                }
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data) + "\n")
                except Exception:
                    pass
                # #endregion
                response = await call_next(request)
                process_time = time.time() - start_time
                print(
                    f"   Request: {request.method} {request.url.path} "
                    f"-> {response.status_code} ({process_time:.3f}s)",
                    flush=True
                )
                # #region agent log
                log_data = {
                    "location": "mcp_service.py:108",
                    "message": "Request completed",
                    "data": {
                        "method": request.method,
                        "path": str(request.url.path),
                        "status": response.status_code,
                        "duration": process_time
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "route-debug",
                    "hypothesisId": "D"
                }
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data) + "\n")
                except Exception:
                    pass
                # #endregion
                return response

            print("✅ Request logging middleware added", flush=True)

            # Log final route count
            if hasattr(app, "routes"):
                print(f"   Final routes count: {len(app.routes)}",
                      flush=True)
                # #region agent log
                final_routes = [
                    {
                        "path": getattr(r, "path", "unknown"),
                        "methods": getattr(r, "methods", [])
                    }
                    for r in app.routes
                ]
                log_data = {
                    "location": "mcp_service.py:120",
                    "message": "Final routes after configuration",
                    "data": {
                        "route_count": len(app.routes),
                        "routes": final_routes
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "route-debug",
                    "hypothesisId": "A"
                }
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data) + "\n")
                except Exception:
                    pass
                # #endregion

            print("✅ App configured", flush=True)
            print(f"   streamable_http_path: "
                  f"{mcp.settings.streamable_http_path}", flush=True)
            print(f"   MCP endpoint: POST "
                  f"{mcp.settings.streamable_http_path}", flush=True)
            print("   Health endpoint: GET /health", flush=True)
            print("   Root endpoint: GET /", flush=True)
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


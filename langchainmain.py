"""
Faculty Search FastAPI Application
==================================
REST API wrapper for the Faculty Search Agent using FastAPI.
Provides endpoints for querying faculty data through natural language.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
import traceback

# Import the faculty search function from your existing code
# Make sure your main faculty search code is in a file named 'faculty_search_agent.py'
# and that the main query function is accessible
try:
    from faculty_search_agent import append_history_and_run
except ImportError as e:
    logging.error(f"Could not import faculty search agent: {e}")
    logging.error("Make sure your main code is in 'faculty_search_agent.py' and properly configured")
    raise

# ================================
# FASTAPI APP SETUP
# ================================

app = FastAPI(
    title="Faculty Search API",
    description="Natural language interface for searching faculty information with LLM-powered agents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "https://vit-bot-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================================
# PYDANTIC MODELS
# ================================

class QueryRequest(BaseModel):
    """Request model for faculty search queries."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Natural language query about faculty information",
        example="Find teachers specialized in AI and machine learning"
    )
    include_debug: Optional[bool] = Field(
        default=False,
        description="Include debug information in response"
    )

class QueryResponse(BaseModel):
    """Response model for faculty search results."""
    success: bool = Field(description="Whether the query was successful")
    query: str = Field(description="The original query")
    response: str = Field(description="The formatted search results")
    processing_time: Optional[float] = Field(default=None, description="Query processing time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if any")
    debug_info: Optional[Dict[str, Any]] = Field(default=None, description="Debug information")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="API health status")
    message: str = Field(description="Health check message")
    version: str = Field(description="API version")
    dependencies: Dict[str, str] = Field(description="Status of dependencies")

class StatsResponse(BaseModel):
    """Statistics response model."""
    total_queries: int = Field(description="Total number of queries processed")
    successful_queries: int = Field(description="Number of successful queries")
    failed_queries: int = Field(description="Number of failed queries")
    uptime: str = Field(description="API uptime")

# ================================
# GLOBAL VARIABLES FOR STATS
# ================================

query_stats = {
    "total": 0,
    "successful": 0,
    "failed": 0
}

import time
start_time = time.time()

# ================================
# UTILITY FUNCTIONS
# ================================

def get_uptime():
    """Calculate API uptime."""
    uptime_seconds = time.time() - start_time
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

def check_dependencies():
    """Check status of dependencies."""
    deps_status = {}
    
    try:
        import pandas
        deps_status["pandas"] = "✓ Available"
    except ImportError:
        deps_status["pandas"] = "✗ Missing"
    
    try:
        import openai
        deps_status["openai"] = "✓ Available"
    except ImportError:
        deps_status["openai"] = "✗ Missing"
        
    try:
        import langchain
        deps_status["langchain"] = "✓ Available"
    except ImportError:
        deps_status["langchain"] = "✗ Missing"
    
    try:
        from faculty_search_agent import append_history_and_run
        deps_status["faculty_agent"] = "✓ Available"
    except ImportError:
        deps_status["faculty_agent"] = "✗ Missing"
    
    return deps_status

# ================================
# API ENDPOINTS
# ================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Faculty Search API with LLM Agents",
        "version": "2.0.0",
        "description": "Natural language interface for searching faculty information",
        "features": [
            "Intelligent teacher search with fuzzy matching",
            "SQL-like queries for structured data analysis",
            "LLM-powered query understanding",
            "Conversation history context"
        ],
        "endpoints": {
            "search": "POST /search - Main search endpoint",
            "search_get": "GET /search/{query} - Simple GET search",
            "health": "GET /health - Health check",
            "stats": "GET /stats - API statistics",
            "docs": "GET /docs - Interactive documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    deps = check_dependencies()
    
    # Check if all critical dependencies are available
    critical_deps = ["pandas", "openai", "langchain", "faculty_agent"]
    all_healthy = all("✓" in deps.get(dep, "") for dep in critical_deps)
    
    status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        message=f"Faculty Search API is {status}",
        version="2.0.0",
        dependencies=deps
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics."""
    return StatsResponse(
        total_queries=query_stats["total"],
        successful_queries=query_stats["successful"],
        failed_queries=query_stats["failed"],
        uptime=get_uptime()
    )

@app.post("/search", response_model=QueryResponse)
async def search_faculty(request: QueryRequest):
    """
    Search faculty information using natural language queries.
    
    This endpoint accepts natural language queries and uses LLM-powered agents
    to understand the query and return relevant faculty information using either:
    - Fuzzy string matching for teacher-specific searches
    - SQL queries for structured data analysis
    
    Examples of queries:
    - "What teachers specialize in machine learning?"
    - "Find professors in Computer Science department"
    - "How many faculty members are in each school?"
    - "Show me Dr. Smith's profile"
    - "List all departments"
    - "Who are the assistant professors in engineering?"
    """
    start_time_query = time.time()
    query_stats["total"] += 1
    
    try:
        logger.info(f"Processing query: '{request.query}'")
        
        # Call the faculty search function with conversation history
        result = append_history_and_run(request.query)
        
        processing_time = time.time() - start_time_query
        query_stats["successful"] += 1
        
        logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
        
        debug_info = None
        if request.include_debug:
            debug_info = {
                "processing_time_seconds": processing_time,
                "query_length": len(request.query),
                "response_length": len(result) if result else 0,
                "timestamp": time.time()
            }
        
        return QueryResponse(
            success=True,
            query=request.query,
            response=result or "No results found.",
            processing_time=round(processing_time, 3),
            debug_info=debug_info
        )
        
    except Exception as e:
        processing_time = time.time() - start_time_query
        query_stats["failed"] += 1
        
        error_msg = str(e)
        logger.error(f"Error processing query: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return QueryResponse(
            success=False,
            query=request.query,
            response="",
            processing_time=round(processing_time, 3),
            error=f"An error occurred while processing your query: {error_msg}"
        )

@app.get("/search/{query}", response_model=QueryResponse)
async def search_faculty_get(query: str, include_debug: bool = False):
    """
    Alternative GET endpoint for simple queries.
    Useful for quick testing or simple integrations.
    
    Parameters:
    - query: The search query (URL encoded)
    - include_debug: Whether to include debug information
    """
    try:
        # Basic validation
        if len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
        
        # Create request object and use the main search function
        request = QueryRequest(query=query, include_debug=include_debug)
        return await search_faculty(request)
        
    except HTTPException:
        raise
    except Exception as e:
        query_stats["total"] += 1
        query_stats["failed"] += 1
        logger.error(f"Error processing GET query: {str(e)}")
        
        return QueryResponse(
            success=False,
            query=query,
            response="",
            error=f"An error occurred while processing your query: {str(e)}"
        )

# ================================
# ADDITIONAL UTILITY ENDPOINTS
# ================================

@app.post("/reset-history")
async def reset_conversation_history():
    """Reset the conversation history."""
    try:
        # Import and clear conversation history
        from faculty_search_agent import conversation_history
        conversation_history.clear()
        
        return {
            "success": True,
            "message": "Conversation history has been reset"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to reset history: {str(e)}"
        }

@app.get("/example-queries")
async def get_example_queries():
    """Get example queries that users can try."""
    return {
        "teacher_search_examples": [
            "Find teachers specialized in machine learning",
            "Who is Dr. John Smith?",
            "Show me professors in the Computer Science department",
            "Find faculty members with PhD in AI"
        ],
        "data_analysis_examples": [
            "How many departments are there?",
            "List all schools",
            "Count faculty by designation",
            "Show me all assistant professors",
            "What are the different areas of specialization?"
        ],
        "mixed_examples": [
            "How many professors work in engineering schools?",
            "Find all faculty with machine learning expertise",
            "Which departments have the most faculty?",
            "Show me recent PhD holders in computer science"
        ]
    }

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {
        "success": False,
        "error": "Endpoint not found",
        "message": "Check /docs for available endpoints",
        "available_endpoints": ["/", "/health", "/search", "/stats", "/docs"]
    }

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    """Handle validation errors."""
    return {
        "success": False,
        "error": "Validation error",
        "message": "Please check your request format",
        "details": str(exc)
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "success": False,
        "error": "Internal server error",
        "message": "Please try again later or contact support"
    }

# ================================
# STARTUP/SHUTDOWN EVENTS
# ================================

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Faculty Search API starting up...")
    
    # Check dependencies on startup
    deps = check_dependencies()
    logger.info("Dependency check:")
    for dep, status in deps.items():
        logger.info(f"  {dep}: {status}")
    
    logger.info("API is ready to accept requests at http://localhost:8000")
    logger.info("Interactive documentation available at http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Faculty Search API shutting down...")
    logger.info(f"Final stats - Total queries: {query_stats['total']}, Successful: {query_stats['successful']}, Failed: {query_stats['failed']}")

# ================================
# MAIN RUNNER
# ================================

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "main:app",  # Change this to match your filename
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

# ================================
# USAGE INSTRUCTIONS
# ================================

"""
SETUP AND USAGE INSTRUCTIONS
=============================

1. File Structure:
   - Save this code as 'main.py'
   - Save your main faculty search code as 'faculty_search_agent.py'
   - Ensure your CSV file 'faculties.csv' is in the same directory
   - Create a '.env' file with your OPENAI_API_KEY

2. Install Dependencies:
   pip install fastapi uvicorn python-dotenv pandas rapidfuzz pandasql langchain-openai langchain

3. Run the Application:
   python main.py
   
   OR
   
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

4. Access the API:
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health
   - Statistics: http://localhost:8000/stats

5. Example API Calls:

   POST /search
   {
     "query": "Find teachers specialized in machine learning",
     "include_debug": false
   }

   GET /search/How%20many%20departments%20are%20there?

6. Response Format:
   {
     "success": true,
     "query": "Find teachers specialized in machine learning",
     "response": "Found 3 results for 'machine learning':\n...",
     "processing_time": 2.341,
     "error": null,
     "debug_info": {...}
   }

7. Production Deployment:
   - Update CORS origins for your domain
   - Use environment variables for configuration
   - Add authentication if needed
   - Configure proper logging
   - Use a production ASGI server like gunicorn
"""
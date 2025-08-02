"""
Faculty Search FastAPI Application
==================================
REST API wrapper for the Faculty Search Agent using FastAPI.
Provides endpoints for querying faculty data through natural language.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import logging

# Import the faculty search function from your existing code
# Make sure the faculty search code is in a file named 'faculty_agent.py'
from faculty_agent import run_query

# ================================
# FASTAPI APP SETUP
# ================================

app = FastAPI(
    title="Faculty Search API",
    description="Natural language interface for searching faculty information",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vit-bot-frontend.vercel.app/"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# PYDANTIC MODELS
# ================================

class QueryRequest(BaseModel):
    """Request model for faculty search queries."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="Natural language query about faculty information",
        example="What teachers specialize in machine learning?"
    )

class QueryResponse(BaseModel):
    """Response model for faculty search results."""
    success: bool = Field(description="Whether the query was successful")
    query: str = Field(description="The original query")
    response: str = Field(description="The formatted search results")
    error: Optional[str] = Field(default=None, description="Error message if any")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="API health status")
    message: str = Field(description="Health check message")

# ================================
# API ENDPOINTS
# ================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Faculty Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Faculty Search API is running successfully"
    )

@app.post("/search", response_model=QueryResponse)
async def search_faculty(request: QueryRequest):
    """
    Search faculty information using natural language queries.
    
    This endpoint accepts natural language queries and returns relevant
    faculty information using either fuzzy string matching or SQL queries.
    
    Examples of queries:
    - "What teachers specialize in machine learning?"
    - "Find professors in Computer Science department"
    - "How many faculty members are in each school?"
    - "Show me Dr. Smith's profile"
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Call the faculty search function
        result = run_query(request.query)
        
        logger.info("Query processed successfully")
        
        return QueryResponse(
            success=True,
            query=request.query,
            response=result
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        
        # Return error response
        return QueryResponse(
            success=False,
            query=request.query,
            response="",
            error=f"An error occurred while processing your query: {str(e)}"
        )

@app.get("/search/{query}", response_model=QueryResponse)
async def search_faculty_get(query: str):
    """
    Alternative GET endpoint for simple queries.
    Useful for quick testing or simple integrations.
    """
    try:
        if len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(query) > 500:
            raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")
        
        logger.info(f"Processing GET query: {query}")
        
        result = run_query(query)
        
        logger.info("GET query processed successfully")
        
        return QueryResponse(
            success=True,
            query=query,
            response=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing GET query: {str(e)}")
        
        return QueryResponse(
            success=False,
            query=query,
            response="",
            error=f"An error occurred while processing your query: {str(e)}"
        )

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {
        "success": False,
        "error": "Endpoint not found",
        "message": "Check /docs for available endpoints"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "success": False,
        "error": "Internal server error",
        "message": "Please try again later"
    }

# ================================
# STARTUP/SHUTDOWN EVENTS
# ================================

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Faculty Search API starting up...")
    logger.info("API is ready to accept requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Faculty Search API shutting down...")

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
To run this FastAPI application:

1. Install required dependencies:
   pip install fastapi uvicorn

2. Make sure your faculty search code is in a file named 'faculty_agent.py'

3. Run the application:
   python main.py
   
   OR
   
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

4. Access the API:
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

5. Example API calls:

   POST /search
   {
     "query": "What teachers specialize in machine learning?"
   }

   GET /search/What%20teachers%20specialize%20in%20machine%20learning?

   Response format:
   {
     "success": true,
     "query": "What teachers specialize in machine learning?",
     "response": "Here are the matching faculty members:\n...",
     "error": null
   }
"""
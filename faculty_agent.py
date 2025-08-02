"""
Faculty Search Agent using LangGraph
=====================================
This application creates an intelligent agent that can search through faculty data
using two different approaches: fuzzy string matching for teacher searches and 
SQL queries for general data analysis.
"""

import pandas as pd
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pandasql import sqldf
from rapidfuzz import fuzz
import os
from dotenv import load_dotenv
load_dotenv()
# ================================
# CONFIGURATION
# ================================

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM with GPT-4o-mini for cost efficiency
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,  # Set to 0 for consistent, deterministic responses
    openai_api_key=OPENAI_API_KEY
)

# ================================
# DATA LOADING
# ================================

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

# Load the faculty dataset
df = load_csv("./faculties.csv")

# ================================
# STATE DEFINITION
# ================================

class AgentState(TypedDict):
    """
    Defines the state structure for the LangGraph agent.
    This state is passed between all nodes in the workflow.
    """
    query: str                                    # User's input query
    tool_choice: Optional[str]                   # Which tool to use (SearchTeacher/QueryCSV)
    search_results: Optional[List[Dict]]         # Results from teacher search
    sql_results: Optional[pd.DataFrame]          # Results from SQL query
    conversation_history: List[str]              # Chat history for context
    response: Optional[str]                      # Final formatted response

# ================================
# WORKFLOW NODES
# ================================

def identify_tool(state: AgentState) -> Dict[str, str]:
    """
    TOOL SELECTION NODE
    
    Analyzes the user query and decides which tool to use:
    - SearchTeacher: For finding specific teachers by attributes
    - QueryCSV: For general data analysis using SQL
    """
    tools = ["SearchTeacher", "QueryCSV"]
    
    prompt = f"""Given the user query below, decide which tool to use.

Available tools:
1. SearchTeacher - For finding specific teachers based on attributes (names, specializations, etc.)
2. QueryCSV - For general data analysis, counting, grouping, or complex filtering using SQL

Query: {state['query']}

Respond with ONLY the tool name (either 'SearchTeacher' or 'QueryCSV')."""

    tool_choice = llm.invoke(prompt).content.strip()
    return {"tool_choice": tool_choice}

def search_teacher(state: AgentState) -> Dict[str, List[Dict]]:
    """
    TEACHER SEARCH NODE
    
    Performs fuzzy string matching to find teachers based on:
    1. Identifies the most relevant column to search in
    2. Extracts meaningful search terms from the query
    3. Uses exact matching (higher priority) and fuzzy matching
    4. Returns ranked results with confidence scores
    """
    query = state['query']
    
    # Step 1: Identify which column to search in
    identify_column_prompt = PromptTemplate(
        input_variables=["query"],
        template='''You are an intelligent assistant tasked with identifying the most relevant column in a CSV file for a given query.

The CSV file contains the following columns:
- School: Institution name
- Department: Academic department
- Name: Faculty member's name
- Designation: Job title/position
- Profile_Link: URL to faculty profile
- Education_Details: Academic background
- Post_Doctoral_Experiences: Research experience
- Areas_of_Specialization: Research areas and expertise

Given the user query: "{query}", determine the single most relevant column to search for the requested information.

Respond with ONLY the exact column name (without any explanations).'''
    )
    
    identify_column_agent = identify_column_prompt | llm
    relevant_column = identify_column_agent.invoke({"query": query}).content.strip()

    # Step 2: Extract meaningful search terms
    meaningful_word_prompt = PromptTemplate(
        input_variables=["query"],
        template='''Extract the most meaningful word or short phrase (2-3 words max) from:
        "{query}"
        
        This should be the key term that best represents what the user is looking for.
        Return ONLY the term exactly as it appears, no explanations.'''
    )
    
    meaningful_word_agent = meaningful_word_prompt | llm
    meaningful_query = meaningful_word_agent.invoke({"query": query}).content.strip()

    # Step 3: Helper function to remove academic titles
    def remove_titles(name: str) -> str:
        """Remove common academic titles from names for better matching."""
        title_prefixes = ['dr.', 'prof.', 'mr.', 'ms.', 'mrs.']
        name_lower = name.lower()
        
        for prefix in title_prefixes:
            if name_lower.startswith(prefix):
                return name[len(prefix):].strip()
        return name

    # Step 4: Prepare data for searching
    entries = df[relevant_column].astype(str)
    entries_without_titles = entries.apply(remove_titles)

    # Step 5: Find exact matches (highest priority)
    start_matches = [
        (i, entry) for i, entry in enumerate(entries_without_titles)
        if entry.lower().startswith(meaningful_query.lower())
    ]

    # Step 6: Find fuzzy matches for remaining entries
    fuzzy_matches = [
        (i, entry, fuzz.partial_ratio(meaningful_query.lower(), entry.lower()))
        for i, entry in enumerate(entries_without_titles)
        if not entry.lower().startswith(meaningful_query.lower())
    ]

    # Step 7: Combine and rank results
    # Exact matches get score of 100
    start_matches_with_scores = [
        {"row": df.iloc[i].to_dict(), "score": 100} 
        for i, entry in start_matches
    ]
    
    # Fuzzy matches with score >= 50 (good confidence threshold)
    filtered_fuzzy_matches = [
        {"row": df.iloc[i].to_dict(), "score": score}
        for i, entry, score in sorted(fuzzy_matches, key=lambda x: x[2], reverse=True)
        if score >= 50
    ]

    # Combine results with exact matches first
    combined_results = start_matches_with_scores + filtered_fuzzy_matches
    
    return {"search_results": combined_results[:7]}  # Return top 7 results

def query_csv(state: AgentState) -> Dict[str, pd.DataFrame]:
    """
    SQL QUERY NODE
    
    Converts natural language queries into SQL and executes them:
    1. Uses LLM to generate appropriate SQL query
    2. Executes query using pandasql
    3. Handles errors gracefully
    """
    query = state['query']
    
    # Generate SQL query using LLM
    prompt = PromptTemplate(
        input_variables=["query"],
        template='''You are an intelligent assistant tasked with generating an SQL query to retrieve relevant data from a database.

### **Database Information:**
- There is **only one table** in the database, and it is named **'df'**.

### **Table Schema:**
The table **'df'** contains the following columns:
- **School**: Institution name
- **Department**: Academic department  
- **Name**: Faculty member's name
- **Designation**: Job title/position
- **Profile_Link**: URL to faculty profile
- **Education_Details**: Academic background
- **Post_Doctoral_Experiences**: Research experience
- **Areas_of_Specialization**: Research areas and expertise

### **Instructions:**
- Generate a valid **SQL SELECT query** based on the user query: **"{query}"**.
- **Return only the SQL query—no explanations or additional text.**
- Use appropriate **WHERE** conditions for filtering.
- Use **LIKE** operator with wildcards (%) for partial text matching.
- If counting or grouping is needed, use appropriate aggregate functions.
- If the query is vague, return a general query selecting relevant columns.

Example patterns:
- For "how many": Use COUNT(*)
- For "list all": Use SELECT with appropriate columns
- For "find teachers who": Use WHERE with LIKE conditions
'''
    )
    
    sql_agent = prompt | llm
    query_text = sql_agent.invoke({"query": query}).content
    
    # Clean up the generated SQL query
    query_text = query_text.replace("```sql", "").replace("```", "").strip()

    # Execute the SQL query
    try:
        result = sqldf(query_text, {"df": df})
        return {"sql_results": result}
    except Exception as e:
        return {"sql_results": pd.DataFrame()}  # Return empty DataFrame on error

def format_response(state: AgentState) -> Dict[str, str]:
    """
    RESPONSE FORMATTING NODE
    
    Takes the results from either search tool and formats them into
    a user-friendly response with proper structure and relevant details.
    """
    if state['tool_choice'] == "SearchTeacher":
        results = state['search_results']
        
        if not results:
            return {"response": "No matching teachers found for your search criteria."}

        response = "Here are the matching faculty members:\n"
        response += "=" * 50 + "\n"
        
        for i, res in enumerate(results, 1):
            teacher = res['row']
            score = res['score']
            
            response += f"\n{i}. **{teacher['Name']}** ({teacher['Designation']})"
            response += f"\n   📍 Department: {teacher['Department']}"
            response += f"\n   🎓 School: {teacher['School']}"
            response += f"\n   🔬 Specializations: {teacher['Areas_of_Specialization']}"
            response += f"\n   🔗 Profile: {teacher['Profile_Link']}"
            response += f"\n   📊 Match Score: {score}%\n"
            
        return {"response": response}

    elif state['tool_choice'] == "QueryCSV":
        results = state['sql_results']
        
        if results.empty:
            return {"response": "No results found for your query. Please try rephrasing your question."}
        
        # Format SQL results nicely
        response = f"Query Results ({len(results)} records found):\n"
        response += "=" * 50 + "\n"
        response += results.to_string(index=False)
        
        return {"response": response}

    return {"response": "I couldn't process your request. Please try again with a different query."}

def update_conversation(state: AgentState) -> Dict[str, List[str]]:
    """
    CONVERSATION HISTORY NODE
    
    Maintains conversation context by storing recent exchanges.
    Keeps only the last 4 messages (2 turns) to manage memory efficiently.
    """
    history = state.get('conversation_history', [])
    
    # Add current exchange to history
    history.append(f"User: {state['query']}")
    history.append(f"Assistant: {state['response']}")
    
    # Keep only last 4 messages (2 complete turns)
    updated_history = history[-4:]
    
    return {"conversation_history": updated_history}

# ================================
# WORKFLOW CONSTRUCTION
# ================================

def build_workflow() -> StateGraph:
    """
    Constructs the LangGraph workflow with all nodes and edges.
    
    Workflow flow:
    1. identify_tool -> decide which tool to use
    2. search_teacher OR query_csv -> process the query
    3. format_response -> create user-friendly output
    4. update_conversation -> maintain chat history
    """
    workflow = StateGraph(AgentState)

    # Add all nodes to the workflow
    workflow.add_node("identify_tool", identify_tool)
    workflow.add_node("search_teacher", search_teacher)
    workflow.add_node("query_csv", query_csv)
    workflow.add_node("format_response", format_response)
    workflow.add_node("update_conversation", update_conversation)

    # Define conditional routing based on tool choice
    workflow.add_conditional_edges(
        "identify_tool",
        lambda x: "search_teacher" if x["tool_choice"] == "SearchTeacher" else "query_csv",
    )

    # Define linear flow after tool execution
    workflow.add_edge("search_teacher", "format_response")
    workflow.add_edge("query_csv", "format_response")
    workflow.add_edge("format_response", "update_conversation")
    workflow.add_edge("update_conversation", END)

    # Set the entry point
    workflow.set_entry_point("identify_tool")

    return workflow

# ================================
# APPLICATION INTERFACE
# ================================

# Build and compile the workflow
workflow = build_workflow()
app = workflow.compile()

def run_query(query: str) -> str:
    """
    Main interface function for running queries through the agent.
    
    Args:
        query (str): User's natural language query
        
    Returns:
        str: Formatted response from the agent
    """
    # Initialize state for this query
    state = {
        "query": query,
        "conversation_history": []
    }
    
    # Execute the workflow
    result = app.invoke(state)
    
    return result['response']

# ================================
# EXAMPLE USAGE
# ================================

# if __name__ == "__main__":
#     # Run a sample query
#     sample_response = run_query("biji c l")
#     print(sample_response)
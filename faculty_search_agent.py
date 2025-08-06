# ============================== #
#        Required Imports         #
# ============================== #
import os
import pandas as pd
from rapidfuzz import fuzz
from dotenv import load_dotenv
from pandasql import sqldf
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
import json
import re
# ============================== #
#        Environment Setup        #
# ============================== #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=OPENAI_API_KEY
)

# ============================== #
#         Load CSV Data           #
# ============================== #
df = pd.read_csv("./faculties.csv")


# ============================== #
#   LLM Prompt Templates Setup    #
# ============================== #
meaningful_word_prompt = PromptTemplate(
    input_variables=["query"],
    template='''You are an intelligent text-processing assistant.

### **Task:**
Extract the **most meaningful word or short phrase** (key term) from the given text.

### **Text:**
"{query}"

### **Instructions:**
- Identify **a single most relevant word or short phrase** (2-3 words max) that best represents the essence of the text.
- **Exclude** stop words, common words (e.g., "the," "is," "and"), and irrelevant terms.
- **Do not modify** or rephrase the extracted term—return it exactly as it appears in the text.
- **Provide only one term (word or phrase) and nothing else**—no explanations, extra text, or formatting.'''
)

identify_column_prompt = PromptTemplate(
    input_variables=["query"],
    template='''You are an intelligent assistant tasked with identifying the most relevant column in a CSV file for a given query.

The CSV file contains the following columns:
- **School**
- **Department**
- **Name**
- **Designation**
- **Profile_Link**
- **Education_Details**
- **Post_Doctoral_Experiences**
- **Areas_of_Specialization**

Given the user query: **"{query}"**, determine the single most relevant column to search for the requested information.

### **Instructions:**
- Respond with **ONLY** the exact column name (without any explanations).
- If multiple columns seem relevant, choose the **most specific one**.
- If the query refers to a person's identity, return **"Name"**.
- If the query is vague, choose the most **general but relevant** column.'''
)

query_csv_prompt = PromptTemplate(
    input_variables=["query"],
    template='''You are an intelligent assistant tasked with generating an SQL query to retrieve relevant data from a database.

### Database Information:
- There is **only one table** in the database, and it is named **df**. This is of High importance do not put any other name of table only df.

### Table Schema:
The table **df** contains the following columns:
- School
- Department
- Name
- Designation
- Profile_Link
- Education_Details
- Post_Doctoral_Experiences
- Areas_of_Specialization

### Important Examples:
1. User Query: "List all schools"
   SQL: SELECT DISTINCT School FROM **df**;

2. User Query: "Show all departments"
   SQL: SELECT DISTINCT Department FROM **df**;

3. User Query: "Find teachers specialized in AI"
   SQL: SELECT * FROM **df** WHERE Areas_of_Specialization LIKE '%AI%';

4. User Query: "Faculty in Computer Science department"
   SQL: SELECT * FROM **df** WHERE Department LIKE '%Computer Science%';

5. User Query: "Professors with designation Assistant Professor"
   SQL: SELECT * FROM **df** WHERE Designation LIKE '%Assistant Professor%';

6. User Query: "Get profile of Dr. John Doe"
   SQL: SELECT * FROM **df** WHERE Name LIKE '%John Doe%';

### Instructions:
- Generate a valid **SQL SELECT query** based on the user query: "{query}".
- Always use **df** as the table name in the FROM clause.
- Return only the SQL query—no explanations or extra formatting.
- ONCE AGAIN REMEMBER ONLY 1 TABLE NAMED DF.
'''
)

verbose_summary_prompt = PromptTemplate(
    input_variables=["raw_data"],
    template='''You are a helpful assistant. Given the following raw data results, provide a detailed, user-friendly explanation. Explain matches, context, and summarize all relevant details nicely.

Raw Data:
{raw_data}

Detailed Explanation:
'''
)

# ============================== #
#      LLM Chains Initialization  #
# ============================== #
meaningful_word_agent = LLMChain(prompt=meaningful_word_prompt, llm=llm)
identify_column_agent = LLMChain(prompt=identify_column_prompt, llm=llm)
query_csv_chain = LLMChain(prompt=query_csv_prompt, llm=llm)
summarizer_chain = LLMChain(prompt=verbose_summary_prompt, llm=llm)

# ============================== #
#     Utility Functions           #
# ============================== #

def remove_titles(name):
    title_prefixes = ['dr.', 'prof.', 'mr.', 'ms.', 'mrs.']
    for prefix in title_prefixes:
        if name.lower().startswith(prefix):
            return name[len(prefix):].strip()
    return name

def search_in_column(df, column_name, query, threshold=50, max_results=7):
    meaningful_query = meaningful_word_agent.run(query)
    entries = df[column_name].astype(str).apply(remove_titles)

    start_matches = [
        (i, entry) for i, entry in enumerate(entries)
        if entry.lower().startswith(meaningful_query)
    ]

    fuzzy_matches = [
        (i, entry, fuzz.partial_ratio(meaningful_query, entry.lower()))
        for i, entry in enumerate(entries)
        if not entry.lower().startswith(meaningful_query)
    ]

    start_matches_with_scores = [{"row": df.iloc[i], "score": 100} for i, _ in start_matches]
    filtered_fuzzy_matches = [
        {"row": df.iloc[i], "score": score}
        for i, _, score in sorted(fuzzy_matches, key=lambda x: x[2], reverse=True)
        if score >= threshold
    ]

    combined_results = start_matches_with_scores + filtered_fuzzy_matches
    return combined_results[:max_results]

def function_searchteacher(query):
    relevant_column = identify_column_agent.run(query)
    results = search_in_column(df, relevant_column, query)

    if not results:
        raw_output = f"No matching teachers found for '{query}'."
    else:
        raw_output = f"Found {len(results)} results for '{query}':\n"
        for idx, result in enumerate(results, 1):
            row = result["row"]
            raw_output += (
                f"{idx}. Name: {row['Name']} | Department: {row['Department']} | "
                f"Designation: {row['Designation']} | Specialization: {row['Areas_of_Specialization']} | "
                f"Profile: {row['Profile_Link']}\n"
            )

    verbose_response = summarizer_chain.run(raw_data=raw_output)
    return verbose_response

def function_querycsvfile(query):
    sql_query = query_csv_chain.run(query)

    try:
        result = sqldf(sql_query, {"df": df})
        if result.empty:
            raw_output = "No results found for the query."
        else:
            raw_output = f"SQL Query Result:\n{result.to_string(index=False)}"
    except Exception as e:
        raw_output = f"SQL Execution Error: {e}"

    # verbose_response = summarizer_chain.run(raw_data=raw_output)
    return raw_output

# ============================== #
#        Tool Wrappers            #
# ============================== #
search_teacher_tool_obj = Tool(
    name="SearchTeacher",
    func=function_searchteacher,
    description="Use this tool when searching for specific teachers based on name, department, designation, or specialization."
)

query_csv_tool_obj = Tool(
    name="QueryCSV",
    func=function_querycsvfile,
    description="Use this tool for structured data analysis on the dataset using SQL-like queries."
)

# ============================== #
#       Router Agent Setup        #
# ============================== #
tools = [query_csv_tool_obj, search_teacher_tool_obj]

decider_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="openai-tools",
    verbose=True,
    max_iterations=5
)

# ============================== #
#     Conversation Flow Logic     #
# ============================== #
conversation_history = []

def append_history_and_run(query):
    global conversation_history

    system_instruction = (
        "You are a helpful assistant. For every response, provide as much detailed information as possible. "
        "Explain search results clearly, add context, and ensure no relevant detail is skipped."
    )

    formatted_query = f"User: {query}"
    conversation_context = "\n".join([f"{q}\n{a}" for q, a in conversation_history])

    full_query = f"System: {system_instruction}\n{conversation_context}\n{formatted_query}" if conversation_context else f"System: {system_instruction}\n{formatted_query}"

    response = decider_agent.run(full_query)

    if len(conversation_history) >= 2:
        conversation_history.pop(0)

    conversation_history.append((formatted_query, f"Assistant: {response}"))
    return response

# ============================== #
#          Example Usage          #
# ============================== #
# if __name__ == "__main__":
#     query1 = "how many DEPARTMENTS are there?"
#     result1 = append_history_and_run(query1)
#     print(result1)

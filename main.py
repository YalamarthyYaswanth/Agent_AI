# main.py — FIXED PROMPT + UPGRADED TO LLAMA3.1
import os
from litellm import completion
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.llms.base import LLM
from typing import Optional, List, Any
from tools.sql_tool import execute_sql_query, get_table_schema, list_tables

# ===========================
# WRAPPER FOR OLLAMA/LLAMA3.1
# ===========================
class OllamaLLM(LLM):
    model: str = "ollama/llama3.1"  # 👈 UPGRADED
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stop=stop,
            api_base="http://localhost:11434"
        )
        return response.choices[0].message.content.strip()

    @property
    def _llm_type(self) -> str:
        return "ollama_llama3.1"

# ===========================
# DEFINE TOOLS
# ===========================
sql_tool = Tool(
    name="SQL_Database",
    func=execute_sql_query,
    description="""
    Executes SQL queries against the PostgreSQL database.
    ALWAYS add WHERE "Freezed" = 'Y' to every query unless user explicitly asks otherwise.
    Input: VALID PostgreSQL SQL query string.
    Main table: collections (in public schema).
    Common columns: "State", "CurrentMonthDemand", "Collections against Current month demand", "Total Overdue Recovery", "Freezed"
    Returns formatted table results.
    """
)

schema_tool = Tool(
    name="Schema_Inspector",
    func=get_table_schema,
    description="Use to get column names and data types for a table. Input: table name (e.g., 'collections')."
)

list_tables_tool = Tool(
    name="List_Tables",
    func=list_tables,
    description="Use to list all tables in 'public' schema. Input: anything (ignored)."
)

# ===========================
# SYSTEM PROMPT — STRICER FORMAT + EXAMPLE
# ===========================
SYSTEM_INSTRUCTION = """
You are CE Analyst Agent. You MUST follow this format EXACTLY:

Thought: [your reasoning]
Action: [tool name]
Action Input: [input to tool]
Observation: [result from tool]
Thought: I now know the final answer
Final Answer: [1-2 sentence summary for user]

📌 RULES:
1. ALWAYS use one of: SQL_Database, Schema_Inspector, List_Tables
2. ALWAYS quote column names in SQL: e.g., "CurrentMonthDemand"
3. ALWAYS add: WHERE "Freezed" = 'Y'
4. If user mentions a state (e.g., 'Maharashtra'), add: AND "State" = '[StateName]'
5. NEVER add state filter if user doesn’t specify state.
6. For Net CE or Gross CE queries, return THREE values:
   - Percentage (Net CE % or Gross CE %)
   - Collection Amount in ₹ Cr (divided by 1e7)
   - Demand Amount in ₹ Cr (divided by 1e7)

📊 CALCULATIONS:

→ Net CE % = (SUM("Collections against Current month demand") * 100.0) / NULLIF(SUM("CurrentMonthDemand"), 0)
→ Net Collection Amount (₹ Cr) = SUM("Collections against Current month demand") / 10000000.0
→ Demand Amount (₹ Cr) = SUM("CurrentMonthDemand") / 10000000.0

→ Gross CE % = (SUM("Collections against Current month demand" + "Total Overdue Recovery") * 100.0) / NULLIF(SUM("CurrentMonthDemand"), 0)
→ Gross Collection Amount (₹ Cr) = SUM("Collections against Current month demand" + "Total Overdue Recovery") / 10000000.0
→ Demand Amount (₹ Cr) = SUM("CurrentMonthDemand") / 10000000.0

💬 SUMMARY FORMAT:
- For Net CE: "[State] achieved Net CE of X%, collecting ₹Y Cr against ₹Z Cr demand."
- For Gross CE: "[State] achieved Gross CE of X%, collecting ₹Y Cr against ₹Z Cr demand."
- If no state: "All-India achieved Net CE of X%, collecting ₹Y Cr against ₹Z Cr demand."

🚫 NEVER:
- Show raw SQL or table to user
- Guess column names — use Schema_Inspector if unsure
- Add state filter unless explicitly mentioned

EXAMPLE 1 (With State):
Question: What is Net CE for n state?
Thought: I need to calculate Net CE, collection amount, and demand amount for Karnataka.
Action: SQL_Database
Action Input: SELECT SUM("Collections against Current month demand") * 100.0 / NULLIF(SUM("CurrentMonthDemand"), 0) AS net_ce_pct, SUM("Collections against Current month demand") / 10000000.0 AS net_coll_cr, SUM("CurrentMonthDemand") / 10000000.0 AS demand_cr FROM collections WHERE "Freezed" = 'Y' AND "State" = 'Karnataka';
Observation: Results:
+----------+--------------+-----------+
| net_ce_pct | net_coll_cr | demand_cr |
+============+==============+===========+
|    x    |    y   |   z |
+------------+--------------+-----------+
Thought: I now know the final answer.
Final Answer: Karnataka achieved Net CE of x%, collecting ₹y Cr against ₹z Cr demand.

EXAMPLE 2 (No State = All-India):
Question: What is Gross CE?
Thought: I need to calculate Gross CE, gross collection amount, and demand amount for all India.
Action: SQL_Database
Action Input: SELECT SUM("Collections against Current month demand" + "Total Overdue Recovery") * 100.0 / NULLIF(SUM("CurrentMonthDemand"), 0) AS gross_ce_pct, SUM("Collections against Current month demand" + "Total Overdue Recovery") / 10000000.0 AS gross_coll_cr, SUM("CurrentMonthDemand") / 10000000.0 AS demand_cr FROM collections WHERE "Freezed" = 'Y';
Observation: Results:
+-------------+---------------+-----------+
| gross_ce_pct | gross_coll_cr | demand_cr |
+==============+================+===========+
|    x   |     y    |  z   |
+--------------+----------------+-----------+
Thought: I now know the final answer.
Final Answer: All-India achieved Gross CE of x%, collecting ₹y Cr against ₹z Cr demand.
"""

# Initialize LLM
llm = OllamaLLM()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define ReAct prompt — NOW INCLUDES {tool_names} ✅
TEMPLATE = '''
You are CE Analyst Agent. Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]  👈 NOW INCLUDED
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
{agent_scratchpad}
'''

prompt = PromptTemplate.from_template(TEMPLATE)

# Create agent — NO CUSTOM PARSER NEEDED WITH LLAMA3.1
agent = create_react_agent(
    llm=llm,
    tools=[sql_tool, schema_tool, list_tables_tool],
    prompt=prompt
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[sql_tool, schema_tool, list_tables_tool],
    memory=memory,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
    early_stopping_method="generate"
)

# ===========================
# AGENT QUERY FUNCTION
# ===========================
def ask_collection_agent(user_query: str) -> str:
    try:
        full_input = f"{SYSTEM_INSTRUCTION}\n\nHuman: {user_query}"
        result = agent_executor.invoke({"input": full_input})
        return result["output"]
    except Exception as e:
        return f"Agent Error: {str(e)}"

# ===========================
# MAIN LOOP
# ===========================
if __name__ == "__main__":
    print("🚀 CE Analyst Agent (LLAMA3.1 + FIXED PROMPT) Ready!")
    print("👉 Ask: 'What is Net CE for Maharashtra?'")
    print("👉 Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        if not user_input:
            continue
            
        print("🧠 Agent is thinking...")
        answer = ask_collection_agent(user_input)
        print(f"🤖 Agent: {answer}\n")
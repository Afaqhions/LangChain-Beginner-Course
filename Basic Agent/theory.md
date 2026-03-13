# LangChain Agents: Theory & Concepts

## What is an Agent?
In LangChain, an **Agent** is a system that uses a Language Model (LLM) as a reasoning engine to determine which actions to take and in what order. 
Unlike standard chains, where the sequence of actions is hardcoded in code, an Agent dynamically decides its path based on the user's input, the context of the conversation, and the available tools.

## Core Components of an Agent

1. **Language Model (LLM)**: This is the "brain" of the agent. The choice of LLM drastically affects how smart and capable the agent is at reasoning, planning, and selecting the right tools.
2. **Tools**: These are functions or capabilities that the agent can use to interact with the outside world. Examples include:
   - Search engines (Tavily, Google, DuckDuckGo)
   - Calculators (Math tools)
   - File system operators
   - Database query tools
   - Custom APIs
   Without tools, the LLM can only answer questions based on its pre-trained knowledge.
3. **Prompt (Agent Instructions)**: The system prompt gives the agent its persona, detailed instructions on how it should behave, the rules it shouldn't break, and the context of what its tools can do.
4. **Agent Executor**: This is the runtime wrapper that manages the agent loop. It takes the output from the agent (which might be "Use tool X with input Y"), actually calls the tool, and then feeds the tool's result (observation) back into the agent so it can decide what to do next.

## How does an Agent work? (The Activity Loop)
A basic agent typically follows a paradigm similar to **ReAct (Reasoning and Acting)**:
1. **Input**: The user asks a question or gives a prompt.
2. **Reasoning (Thought)**: The LLM analyzes the question, thinks about its current knowledge and tools, and determines if it needs an external tool to solve the problem.
3. **Action (Tool Call)**: If a tool is needed, the LLM decides which tool to use and what input to provide to it.
4. **Observation**: The Agent Executor runs the external tool and returns the raw output (observation) back to the LLM.
5. **Review**: The LLM reviews the observation. If the data is sufficient to fulfill the user's goal, it proceeds to formulate an answer. If not, it goes back to Step 2 (Reasoning) and may trigger another tool call.
6. **Final Output**: The agent formulates and returns the final response to the user.

## Why use Agents?
- **Dynamism**: They can handle complex, multi-step tasks where the sequence of steps isn't known ahead of time.
- **Extensibility**: You can exponentially increase your LLM's capabilities simply by adding more custom tools into its toolkit.
- **Overcoming Limitations**: By connecting an LLM to a search tool or an internal database, it overcomes its knowledge cut-off date and hallucinations problem.

## Common Agent Types in LangChain
- **Tool Calling Agent (e.g., OpenAI Tools Agent)**: Utilizes native function/tool calling capabilities built directly into models (like OpenAI, Anthropic, Gemini). Highly recommended because the models are fine-tuned directly for this.
- **ReAct Agent**: A prompt-based approach guiding the model to explicitly output "Thought:", "Action:", and "Observation:". Great for legacy or open-source models that don't support native tool calling yet.
- **Structured Chat Agent**: Designed to be used with Chat Models for handling tools that require complex, structured multiple inputs.

## Building a Simple Agent (Code Blueprint)

Here is a conceptual look at how you transition from a standard Chat Model invocation (like `model.invoke()`) to an Agent:

### 1. Define your Tools
Tools are just Python functions decorated with `@tool`. LangChain uses the function's docstring to understand *when* to use it.

```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Returns the weather for a given location."""
    # In reality, this would call a weather API
    return f"The weather in {location} is sunny and 75°F."

tools = [get_weather]
```

### 2. Initialize the Model and Bind Tools
Agents require models that support tool calling. We bind the tools to the model so the model knows they exist.

```python
from langchain.chat_models import init_chat_model

# Initialize the model as you normally would
model = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq"
)

# Bind the tool to the model
model_with_tools = model.bind_tools(tools)
```

### 3. Create the Agent
LangGraph provides a pre-built `create_react_agent` that simplifies the ReAct and Tool Calling loop.

```python
from langgraph.prebuilt import create_react_agent

# Pass in the model and the list of tools
agent_executor = create_react_agent(model, tools)
```

### 4. Run the Agent
Instead of a simple string output, the agent will execute tools if necessary and return a structured response.

```python
response = agent_executor.invoke({"messages": [("user", "What is the weather in Paris?")]})

# The response will contain the full conversation flow, 
# including the model's decision to call `get_weather` and the final answer.
print(response["messages"][-1].content)
```

### Summary of the Flow:
1. User -> `agent_executor.invoke(...)`
2. Agent -> "I need to know the weather in Paris. I'll use the `get_weather` tool."
3. Agent Executor -> Calls `get_weather("Paris")`
4. Tool -> Returns "sunny and 75°F"
5. Agent -> "The weather in Paris is sunny and 75°F." -> User

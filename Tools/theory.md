# Tools in LangChain: Theory & Integration

## What are Tools?
In LangChain, a **Tool** is simply an interface that the Language Model (LLM) uses to interact with external systems. 
While an LLM can generate text and reason, its knowledge is limited by its training data cutoff and it lacks the ability to take tangible actions. Tools bridge this gap by giving models the abilities to:
- Browse the internet (Search)
- Run code (Python Read-Evaluate-Print Loop)
- Perform math calculations (Calculators)
- Query databases (SQL connectors)
- Fetch weather, fetch stock prices, API integration, etc.

## 1. Defining a Tool
The easiest way to create a tool in LangChain is by decorating a Python function with the `@tool` decorator.

```python
from langchain.tools import tool

@tool
def add(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."
```
> **Crucial Detail**: The docstring (`"""Add two numbers together."""`) and the variable types (`x: int, y: int`) are incredibly important! LangChain extracts these and feeds them to the LLM so it knows *when* to use the tool and *what arguments* to pass to it.

## 2. Binding Tools to the LLM (Tool Calling)
Before a model can use your tools, it must be aware they exist. This is done by *binding* the tools to your model.

```python
# Assuming 'model' is already initialized
tools = [add, get_weather]
model_with_tools = model.bind_tools(tools)
```
When you invoke `model_with_tools`, the LLM doesn't actually run the Python function. Instead, it outputs a special response called `tool_calls` containing:
- The `name` of the tool it wants you to run.
- The `args` (arguments) it wants you to pass to the tool.

## 3. The Tool Execution Loop (Under the Hood)
When a model asks to use a tool, you are responsible for running the code and giving the response back to the model. Here is what happens under the hood when you build the execution loop manually:

1. **Invoke the Model**: The user asks "What is 2 + 2?". The model looks at its bound tools and responds with a `tool_call` request for the `add` tool.
2. **Execute the Tool**: You extract the `tool_call` from the AI's response, find the matching Python function, and invoke the function using the provided arguments.
3. **Return the Result**: You take the result of the function (`4`), wrap it up in a `ToolMessage`, and append it to the conversation history.
4. **Final Model Response**: You prompt the model again with the new `ToolMessage` included in the history. The model sees the result (`4`) and formats a nice human-readable response: "The sum of 2 + 2 is 4."

## 4. Automating with Agents
Manually writing loops to extract tool calls, invoke functions, and feed them back to the model can be tedious and complex, especially for multi-step reasoning.
An **Agent** handles this entire execution loop for you.

```python
from langchain.agents import create_agent
# Alternatively, with LangGraph: from langgraph.prebuilt import create_react_agent

agent = create_agent(
    model=model,
    tools=[add, get_weather],
    system_prompt="You are a helpful assistant."
)

# The agent automatically determines when to call a tool, calls it, and processes the answer!
res = agent.invoke({"messages": [
    {"role": "user", "content": "What is the weather like in Karachi?"}
]})
```

By allowing Agents to manage tools, your application becomes highly dynamic and capable of routing complex requests smoothly.

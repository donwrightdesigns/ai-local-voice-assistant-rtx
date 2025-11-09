"""
LangChain agent setup for voice assistant.
Handles agent creation, memory, and tool integration.
"""

# Compatibility imports across LangChain versions
try:
    from langchain.agents import AgentExecutor, create_react_agent  # Newer APIs
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.tools import Tool
    _LC_EXECUTOR_AVAILABLE = True
except Exception:
    # Older LC: no AgentExecutor available; we'll fall back to ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.tools import Tool
    try:
        from langchain.chains import ConversationChain
    except Exception:
        ConversationChain = None
    create_react_agent = None
    _LC_EXECUTOR_AVAILABLE = False
import asyncio


def create_agent(llm, system_prompt: str = None):
    """
    Create a conversational agent with memory.
    Uses ReAct agent when AgentExecutor is available, otherwise falls back to ConversationChain.
    """
    if system_prompt is None:
        system_prompt = """You are a helpful voice assistant named Luna. 
You provide concise, friendly responses (under 50 words when possible).
You reason through problems step-by-step using available tools.
Always be direct and honest."""

    # Memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # Define tools (extensible)
    tools = [
        Tool(
            name="get_time",
            func=lambda _: __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            description="Get the current date and time",
        ),
    ]

    if _LC_EXECUTOR_AVAILABLE and create_react_agent is not None:
        # ReAct prompt template
        prompt = PromptTemplate.from_template(
            """
You are Luna, a helpful voice assistant.

{chat_history}

You can use the following tools:
{tool_names}

Tool descriptions:
{tools}

Question: {input}

{agent_scratchpad}
"""
        )
        # Create agent graph and executor
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )
        return executor

    # Fallback: simple conversation chain with memory
    if ConversationChain is None:
        # Last-resort minimal shim that echoes
        class _Shim:
            def invoke(self, inputs: dict):
                return {"output": "I'm not fully initialized. Please install a recent langchain."}
        return _Shim()

    prompt = PromptTemplate.from_template(
        """
You are Luna, a concise and helpful assistant.
{chat_history}
Human: {input}
Assistant:
"""
    )
    chain = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)

    class _ChainAdapter:
        def __init__(self, chain):
            self._chain = chain
        def invoke(self, inputs: dict):
            text = inputs.get("input", "")
            out = self._chain.predict(input=text)
            return {"output": out}

    return _ChainAdapter(chain)


def build_langchain_llm(provider_name: str, config: dict):
    """
    Build LangChain LLM from provider config.
    
    Args:
        provider_name: "openai", "ollama", or "vllm"
        config: Provider configuration dict
        
    Returns:
        LangChain LLM instance
    """
    
    if provider_name == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.get("model", "gpt-4o-mini"),
            temperature=0.7,
            streaming=True
        )
    
    elif provider_name == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=config.get("model", "llama3.2:3b"),
            base_url=config.get("base_url", "http://localhost:11434"),
            temperature=0.7,
        )
    
    elif provider_name in ("vllm", "local"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=config.get("endpoint", "http://localhost:8000/v1"),
            api_key="not-needed",
            model="local",
            temperature=0.7,
            streaming=True
        )
    
    raise ValueError(f"Unknown provider: {provider_name}")

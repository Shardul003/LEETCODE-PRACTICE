from langchain.agents import initialize_agent, load_tools
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-turbo")

# Load calculator tool
tools = load_tools(["llm-math"], llm=llm)

# Create an agent that can use the tool
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

# Ask a question
result = agent.run("If I save 500 every month for 2 years, how much will I have?")
print(result)

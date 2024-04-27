
###############
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Include the LLM from a previous lesson
from llm import llm

from langchain.tools import Tool


from agent import generate_response


tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
        )
]

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

from langchain.prompts import PromptTemplate

agent_prompt = PromptTemplate.from_template("""
You are a twitter expert providing information about tweets.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to tweets, users, followers, folowings.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

# agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
    );

def generate_response(prompt):
    response = agent_executor.invoke({"input": prompt})
    return response['output']

##########################

import streamlit as st
from utils import write_message



# tag::setup[]
# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")
# end::setup[]

# tag::session[]
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the TwitterChatBot!  How can I help you?"},
    ]
# end::session[]

# tag::submit[]
# Submit handler

# Submit handler
def handle_submit(message):
    # Handle the response
    with st.spinner('Thinking...'):

        response = generate_response(message)
        write_message('assistant', response)
# end::submit[]


# tag::chat[]
# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
# end::chat[]



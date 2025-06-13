import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

# === HARD-CODED GEMINI API KEY ===
GOOGLE_API_KEY = "AIzaSyDK5hw1kVgVBgJJc64SHH7T9pJOWM2U2lk "  # Replace this with your actual key
# === INITIALIZE GEMINI CHAT MODEL ===
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.5)

# === SETUP DUCKDUCKGO TOOL ===
search_tool = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search_tool.run,
        description="Useful for answering questions about current events or facts from the web."
    )
]

# === INITIALIZE AGENT ===
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False  # Turn off verbose logging
)

# === STREAMLIT UI ===
st.set_page_config(page_title="📰 Real-Time Q&A with Gemini", page_icon="🔍")
st.title("🔍 Ask About Current Events with Gemini + DuckDuckGo")

st.markdown("Type any **question** below and let Gemini + DuckDuckGo give you the answer in real time! 🌐")

query = st.text_input("💬 Your Question:")
ask_btn = st.button("🚀 Ask")

if ask_btn and query.strip():
    try:
        with st.spinner("Thinking... 🤖"):
            response = agent.run(query)
        st.success("✅ Here's the answer:")
        st.write(response)
    except Exception as e:
        st.error(f"⚠️ Oops! Something went wrong.\n\n`{str(e)}`")
elif ask_btn:
    st.warning("❗ Please enter a question before hitting 'Ask'.")


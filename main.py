from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
import streamlit as st
from streamlit_feedback import streamlit_feedback
import time
import os


st.set_page_config(page_title="Self Learning GPT", page_icon="🦜")
st.title("🦜 Self Learning GPT")


def check_if_key_exists(key):
    return key in st.session_state


# API KEY 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# secrets.toml 파일에 저장된 API KEY를 사용할 때
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

if "query" not in st.session_state:
    st.session_state.query = None

reset_history = st.sidebar.button("대화내용 초기화", type="primary")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API KEY", type="password")
    langchain_api_key = st.text_input("LangSmith API KEY", type="password")

    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key
    if langchain_api_key:
        st.session_state["langchain_api_key"] = langchain_api_key

    project_name = st.text_input("LangSmith 프로젝트", value="SELF_LEARNING_GPT")
    session_id = st.text_input("세션 ID(선택사항)")

if not check_if_key_exists("langchain_api_key"):
    st.info(
        "⚠️ [LangSmith API key](https://python.langchain.com/docs/guides/langsmith/walkthrough) 를 추가해 주세요."
    )
else:
    langchain_endpoint = "https://api.smith.langchain.com"
    # LangSmith 설정
    client = Client(
        api_url=langchain_endpoint, api_key=st.session_state["langchain_api_key"]
    )
    ls_tracer = LangChainTracer(project_name=project_name, client=client)
    run_collector = RunCollectorCallbackHandler()
    cfg = RunnableConfig()
    cfg["callbacks"] = [ls_tracer, run_collector]

if not check_if_key_exists("openai_api_key"):
    st.info(
        "⚠️ [OpenAI API key](https://platform.openai.com/docs/guides/authentication) 를 추가해 주세요."
    )


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# 메모리
msgs = StreamlitChatMessageHistory(key="langchain_messages")


if reset_history:
    msgs.clear()
    # msgs.add_ai_message("무엇을 도와드릴까요?")
    st.session_state["last_run"] = None
    st.session_state.messages = []
    st.session_state.query = None


if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

with st.sidebar:
    instructions = st.text_area("지시사항", value="한글로 간결하게 답변하세요")


# 유저의 입력을 받아서 대화를 진행합니다.
if user_input := st.chat_input():
    if check_if_key_exists("openai_api_key") and check_if_key_exists(
        "langchain_api_key"
    ):
        cfg["configurable"] = {"session_id": session_id}
        if st.session_state.query is None:
            st.session_state.query = user_input
            cfg["metadata"] = {"query": user_input}
        else:
            cfg["metadata"] = {"query": st.session_state.query}
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            llm = ChatOpenAI(
                streaming=True,
                callbacks=[stream_handler],
                api_key=st.session_state["openai_api_key"],
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", instructions),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]
            )
            chain = prompt | llm
            chain_with_history = RunnableWithMessageHistory(
                chain,
                lambda session_id: msgs,
                input_messages_key="question",
                history_messages_key="history",
            )
            response = chain_with_history.invoke({"question": user_input}, cfg)
            st.session_state.messages.append(
                ChatMessage(role="assistant", content=response.content)
            )
        wait_for_all_tracers()
        st.session_state.last_run = run_collector.traced_runs[0].id


@st.cache_data(ttl="2h", show_spinner=False)
def get_run_url(run_id):
    time.sleep(1)
    return client.read_run(run_id).url


if st.session_state.get("last_run"):
    run_url = get_run_url(st.session_state.last_run)
    st.sidebar.markdown(f"[LangSmith 추적🛠️]({run_url})")
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label=None,
        key=f"feedback_{st.session_state.last_run}",
    )
    if feedback:
        scores = {"👍": 1, "👎": 0}
        client.create_feedback(
            st.session_state.last_run,
            feedback["type"],
            score=scores[feedback["score"]],
            comment=st.session_state.query,
        )
        st.toast("피드백을 저장하였습니다.!", icon="📝")

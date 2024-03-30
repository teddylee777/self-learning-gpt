from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
import streamlit as st
import os


st.set_page_config(page_title="Self Learning GPT 테스트", page_icon="🦜")
st.title("🦜 Self Learning GPT 테스트")

# API KEY 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# secrets.toml 파일에 저장된 API KEY를 사용할 때
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]


def check_if_key_exists(key):
    return key in st.session_state


if not check_if_key_exists("langchain_api_key"):
    st.info(
        "⚠️ [LangSmith API key](https://python.langchain.com/docs/guides/langsmith/walkthrough) 를 추가해 주세요."
    )
else:
    # LangSmith 설정
    client = Client(api_key=st.session_state["langchain_api_key"])


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def create_qa_pairs(dataset_id, session_id=None):
    qa_pairs = []
    examples = client.list_examples(dataset_id=dataset_id)
    for exp in examples:
        qa = dict()
        ret = extract_qa_from_runid(exp.source_run_id, session_id)

        if ret is None:
            continue
        qa["question"] = ret["question"]
        qa["answer"] = ret["answer"]
        qa["session_id"] = ret["session_id"]
        qa_pairs.append(qa)
    return qa_pairs


def extract_qa_from_runid(run_id, session_id=None):
    ret = dict()
    filter_flag = True
    try:
        for k, v in client.read_run(run_id):
            if k == "outputs":
                ret["answer"] = v["output"]["content"]

            if k == "extra":
                ret["question"] = v["metadata"]["query"]
                if session_id is not None:
                    sid = v["metadata"]["session_id"]
                    if sid.strip() == session_id.strip():
                        ret["session_id"] = sid.strip()
                    else:
                        filter_flag = False
                elif "session_id" in v["metadata"]:
                    ret["session_id"] = v["metadata"]["session_id"].strip()
                else:
                    ret["session_id"] = ""
        if filter_flag:
            return ret
        else:
            return None
    except Exception as e:
        print(e)
        return None


reset_history = st.sidebar.button("대화내용 초기화", type="primary")
if reset_history:
    st.session_state["last_run"] = None
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

with st.sidebar:
    dataset_name = st.text_input("Dataset name 입력(필수)")
    session_id = st.text_input("세션 ID(선택사항)")

    dataset_btn = st.button("데이터셋 로드")
    instructions = st.text_area("지시사항", value="한글로 간결하게 답변하세요")

    if dataset_btn:
        if check_if_key_exists("langchain_api_key"):
            # LangSmith 설정
            st.session_state.examples = client.list_datasets(dataset_name=dataset_name)

            datasets = client.list_datasets(dataset_name=dataset_name)
            session_id = None if session_id.strip() == "" else session_id

            try:
                dataset = next(iter(datasets))
                qa_pairs = create_qa_pairs(dataset.id, session_id=session_id)
                st.session_state.examples = qa_pairs
                st.markdown(
                    f"`{dataset.name}`: `{dataset.id}`, 총 {len(qa_pairs)} 개 QA 세트 로드 완료!"
                )

            except StopIteration:
                st.write("데이터셋이 없습니다.")
        else:
            st.info(
                "⚠️ [LangSmith API key](https://python.langchain.com/docs/guides/langsmith/walkthrough) 를 추가해 주세요."
            )


# 유저의 입력을 받아서 대화를 진행합니다.
if user_input := st.chat_input():
    if check_if_key_exists("openai_api_key"):
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
            if "examples" not in st.session_state:
                st.session_state.examples = []

            example_prompt = PromptTemplate(
                input_variables=["instruction", "question", "answer"],
                template="Question: {question}\nResponse: {answer}",
            )
            qa_pairs = st.session_state.examples
            prompt = FewShotPromptTemplate(
                examples=qa_pairs,
                example_prompt=example_prompt,
                suffix="----\nQuestion: {question}\nResponse: ",
                input_variables=["question"],
            )

            final_prompt = PromptTemplate.from_template(
                """Please refer to the INSTRUCTIONS and FEW SHOT examples to answer the question.
    #INSTUCTIONS:
    {instruction}

    #FEW EXAMPLES:
    {few_shot}                             
    """
            )

            chain = final_prompt | llm

            response = chain.invoke(
                {
                    "few_shot": prompt.format(question=user_input),
                    "instruction": instructions,
                }
            )
            st.session_state.messages.append(
                ChatMessage(role="assistant", content=response.content)
            )
    else:
        st.info(
            "⚠️ [OpenAI API key](https://platform.openai.com/docs/guides/authentication) 를 추가해 주세요."
        )

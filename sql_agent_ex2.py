"""
few-shot 쿼리 예시를 임베딩해두고 agent를 이용하여 결과 뽑아오기
"""

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool

import os
from dotenv import load_dotenv
load_dotenv()

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name='gpt-3.5-turbo',
    #model_name='gpt-4-0125-preview',
    temperature=0,
    verbose=True
)
embeddings = OpenAIEmbeddings()

"""
사용자 질의에 대응하는 쿼리를 few shot으로 만들어 벡터 디비에 임베딩 해두기
"""
few_shots = {
    "List all artists": "SELECT * FROM artists;",
    "Find the total duration of all tracks": "SELECT SUM(Milliseconds) FROM tracks;",
    "How many employees are there": "SELECT COUNT(*) FROM employee;"
}

few_shot_docs = [
    Document(page_content=f"${question} : ${few_shots[question]}", metadata={"sql_query": few_shots[question]})
    for question in few_shots.keys()
]

vector_db = FAISS.from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever()

"""
LLM이 아래 설명을 읽고 필요 시, 활용할 수 있도록 tool을 제공하기
"""
tool_description = """
이 도구는 유사한 예시를 이해하여 사용자 질문에 적용하는 데 도움이 됩니다.
이 도구에 입력하는 내용은 사용자 질문이어야 합니다.
"""
retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)
custom_tool_list = [retriever_tool]


"""
"""

custom_suffix = """
가장 먼저 제가 알고 있는 비슷한 예제를 가져와야 합니다.
예제가 쿼리를 구성하기에 충분하다면 쿼리를 작성할 수 있습니다. 예제의 metadata 중 sql_query에 적힌 쿼리를 참고하세요.
그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS, # openai에서 제공하는 타입에 맞춰서 설정
    extra_tools=custom_tool_list,          # 앞서 만든 retriver tool을 넣어줌
    suffix=custom_suffix                   # LLM에 들어갈 Prompt의 첫번째 부분을 변경주고자 custom_suffix를 사용
)

agent.invoke("How many employees do we have?")

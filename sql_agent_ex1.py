"""
Agent가 질문에 대한 결과를 도출하기 위해,
무엇을 해야하는지 단계적으로 생각하여 계속해서 LLM과 상호작용함으로써
결과를 뽑아내는 방식
"""

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

import os
from dotenv import load_dotenv
load_dotenv()

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name='gpt-3.5-turbo',
    temperature=0,
    verbose=True
)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION # no example
)

#agent_executor.invoke("국가별 총 매출을 나열합니다. 어느 국가의 고객이 가장 많이 지출했나요?")
agent_executor.invoke("playlisttrack 테이블에 대해서 설명해줄래?")


"""
주어진 질의에 맞는 query를 만들어 실행 결과까지 받아오는 SQLDatabaseChain 사용 예제
- 쿼리를 만들어서 검증하지 않은 채 알아서 실행하는 구조라 SQL 인젝션에 취약
"""

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

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
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

db_chain.invoke({"query": "How many employees are there?"})

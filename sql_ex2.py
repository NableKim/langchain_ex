"""
주어진 질의에 맞는 query만 생성하기 위해  create_sql_query_chain를 사용하는 예제
다시 말해서, 쿼리만 만들어주므로 쿼리 실행은 별도로 해야함
"""

from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

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
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there?"})
print(response)

print(db.run(response))
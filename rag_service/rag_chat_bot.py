from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # 큰 문서의 덩어리를 작게 분할하는 과정
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import key
import os

documents = TextLoader('AI.txt').load()


def split_docs(documents, chunk_size=1000, chunk_overlap=20): # 청크 사이즈에 따라 다큐먼트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc = text_splitter.split_documents(documents)
    return doc


docs = split_docs(documents)
print('청크 나누기 완료')

embeddings = embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# chromdb 에 벡터 저장

db = Chroma.from_documents(docs, embeddings)
print('벡터 저장 완료')
os.environ["OPENAI_API_KEY"] = key.open_ai_key


model_name = 'gpt-3.5-turbo'

llm = ChatOpenAI(model_name=model_name)

# Q&A 체인을 사용하여 쿼리에 대한 답변 얻기
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
query = "AI란?"
matching_docs = db.similarity_search(query)
print('db 에서 찾기 완료')
answer = chain.run(input_documents=matching_docs, question=query)
print(answer)




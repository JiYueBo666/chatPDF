import os
import openai
from openai import OpenAI
from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


_=load_dotenv(find_dotenv())

openai_api_key=os.environ['openai_api_key']
base_url=os.environ['base_url']


def get_embedding(text):
    client=OpenAI(api_key=openai_api_key,base_url=base_url)
    response = client.embeddings.create(model="text-embedding-ada-002",
    input=text
    )
    return response.data[0].embedding

class PDFQuery:
    def __init__(self) -> None:
        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key,base_url=base_url)
        self.llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key,
                 base_url=base_url)
        self.chain=None
        self.db=None
    
    def ask(self,question:str)->str:
        if self.chain is None:
            response="请添加一个文件"
        else:
            docs=self.db.get_relevant_documents(question)
            print(docs)
            print(len(docs),'----------------------')
            response=self.chain.run(input_documents=docs,question=question)
        return response
    def ingest(self,file_path:os.PathLike)->None:
        loader=PyPDFium2Loader(file_path)
        documents=loader.load()
        splitted_documents=self.text_splitter.split_documents(documents)

        self.db=Chroma.from_documents(splitted_documents,
        self.embeddings).as_retriever()

        self.chain=load_qa_chain(ChatOpenAI(temperature=0,openai_api_key=openai_api_key,
        base_url=base_url),chain_type="stuff")

    def forget(self)->None:
        self.db=None
        self.chain=None

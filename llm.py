import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["openai_api_key"]
base_url = os.environ["base_url"]



class PDFQuery:
    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=50
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, base_url=base_url,
            model="text-embedding-ada-002",
        )
        self.llm = ChatOpenAI(
            temperature=0, openai_api_key=openai_api_key, base_url=base_url
        )
        self.chain = None
        self.db = None
        self.bm25_retriever = None
        self.retriever_contents=[]

        self.prompt_template = """
        你是一个PDF问答机器人，你的任务是回答用户的问题。
        你需要根据用户的问题，从以下上下文中找到答案。
        如果无法从上下文中找到答案，你需要说“我不知道”。
        请记住，你只能回答用户的问题，不能生成新的内容。
        ___________________
        下面是用户的问题:
        {question}
        ___________________

        下面是检索到的相关文档，作为你的参考：
        ___________________________________        
        1.{doc1}

        2.{doc2}

        3.{doc3}

        4.{doc4}

        5.{doc5}        

        6.{doc6}
        ____________________________________
        如果你认为检索到的文档与用户问题无关，则回答：
        “没有检索到相关内容呢”
        """


    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "请添加一个文件"
        else:
            if self.ensemble_retriever:
                docs=self.ensemble_retriever.get_relevant_documents(question)
            else:
                docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        if docs is not None:
            for i,doc in enumerate(docs):
                content=f"检索到的第{i}个上下文：{doc.page_content}\n"
                self.retriever_contents.append(content)

        prompt=PromptTemplate.from_template(self.prompt_template)
        template_value = {
            'question': question,
            'doc1': docs[0].page_content if len(docs) > 0 else "",
            'doc2': docs[1].page_content if len(docs) > 1 else "",
            'doc3': docs[2].page_content if len(docs) > 2 else "",
            'doc4': docs[3].page_content if len(docs) > 3 else "",
            'doc5': docs[4].page_content if len(docs) > 4 else "",
            'doc6': docs[5].page_content if len(docs) > 5 else ""
        }
        #prompt.invoke(template_value)

        chain=prompt | self.llm |StrOutputParser()
        result=chain.invoke(template_value)
        return result

        #return response

    def ingest(self, file_path: os.PathLike) -> None:
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)

        self.db = Chroma.from_documents(
            splitted_documents, self.embeddings
        ).as_retriever(search_kwargs={"k": 3})

        #创建BM2.5检索器
        self.bm25_retriever = BM25Retriever.from_documents(splitted_documents)
        self.bm25_retriever.k=3#设置文档返回数量

        self.ensemble_retriever =  EnsembleRetriever(
            retrievers=[self.db, self.bm25_retriever],
            weights=[0.4, 0.6],
        )

        self.chain = load_qa_chain(
            ChatOpenAI(temperature=0, openai_api_key=openai_api_key, base_url=base_url),
            chain_type="stuff",
        )

    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.documents = None

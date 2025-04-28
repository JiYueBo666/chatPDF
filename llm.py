from ast import For
from langchain_core.documents.base import Document
import os
import re

from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever,TFIDFRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
import jieba

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["openai_api_key"]
base_url = os.environ["base_url"]

pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)

class PDFQuery:
    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50,
        separators = [r"第\S*条 "],
        is_separator_regex = True
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

    def search_doc(self, question: str) -> list[Document]:
        #分词
        sub_words = self.cut_words(question)
        
        #基于关键词检索文档
        docs_search_by_key_words=[]
        
        for key_words in sub_words:
            docs_key_words = self.bm25_retriever.invoke(key_words)  # 使用 invoke 替代 get_relevant_documents
            
            if len(docs_key_words) > 1:
                docs_search_by_key_words.extend(docs_key_words)
    
        # 基于混合检索文档
        if self.ensemble_retriever:
            docs = self.ensemble_retriever.invoke(question)  # 使用 invoke 替代 get_relevant_documents
        else:
            docs = self.db.invoke(question)  # 使用 invoke 替代 get_relevant_documents
        
        # 合并关键词检索和语义检索的结果
        combined_docs = []
        seen_contents = set()
        
        # 首先添加语义检索结果
        for doc in docs:
            if doc.page_content not in seen_contents:
                combined_docs.append(doc)
                seen_contents.add(doc.page_content)
        # 添加关键词检索结果
        for doc in docs_search_by_key_words:
            if doc.page_content not in seen_contents:
                combined_docs.append(doc)
                seen_contents.add(doc.page_content)
        return combined_docs  

    def ask(self, docs,question) -> str:
        if docs is not None:
            for i,doc in enumerate(docs):
                self.retriever_contents.append(doc.page_content)
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
        chain=prompt | self.llm |StrOutputParser()
        result=chain.invoke(template_value)
        return result

    def cut_words(self,query):
        sub_words=[]
        words = jieba.cut(query)
        for word in words:
            sub_words.append(word)
        return sub_words

    def ingest(self, file_path: os.PathLike) -> None:
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
    
        # 清洗文档
        for i ,doc in enumerate(splitted_documents):
            doc.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''),
        doc.page_content)
    
        # 创建 TFIDFRetriever
        self.tfidf_retriever = TFIDFRetriever.from_documents(splitted_documents)
    
        self.db = Chroma.from_documents(
            splitted_documents, self.embeddings
        ).as_retriever(search_kwargs={"k": 2})
    
        # 创建BM2.5检索器
        self.bm25_retriever = BM25Retriever.from_documents(splitted_documents)
        self.bm25_retriever.k=2#设置文档返回数量
    
        self.ensemble_retriever =  EnsembleRetriever(
            retrievers=[self.db, self.bm25_retriever, self.tfidf_retriever],
            weights=[0.33, 0.33, 0.33],  # 调整权重
        )
    

    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.documents = None

model=PDFQuery()

path=r"D:\AIProject\chatPDF\data\中华人民共和国消费者权益保护法.pdf"


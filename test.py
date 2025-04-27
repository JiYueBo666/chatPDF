import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter,NLTKTextSplitter

file_path = r"data\中华人民共和国消费者权益保护法.pdf"  # Replace with the actual file path

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50
)

loader = PyPDFium2Loader(file_path)
documents = loader.load()
splitted_documents = text_splitter.split_documents(documents)
import re
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)


for i ,doc in enumerate(splitted_documents):
    doc.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''),
    doc.page_content)
    print(f"第{i}段：")
    print(doc.page_content)
    print("-------------------")

    if i>4:
        break
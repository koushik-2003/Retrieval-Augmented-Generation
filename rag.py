import os
import time
from google.colab import userdata
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
load_dotenv()
os.environ['GROQ_API_KEY'] = "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"



def process_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    print("data loading.....")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    print("splitting text....")
    docs = text_splitter.split_documents(data)

    embedding = HuggingFaceEmbeddings()

    vectorstore = FAISS.from_documents(docs, embedding)
    print("indexing.....")
    time.sleep(2)
    return vectorstore
def querying_index(query,vectorstore):
    llm = ChatGroq(temperature=0.9)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain.invoke({"question":query})
    return result

def main():
    urls = ["https://und.edu/"]
    process_urls_flag = input("yes/no").strip().lower()
    if process_urls_flag=="yes":
        vectorstore = process_urls(urls)
        print("Processing completed. FAISS index created.")
    while(True):
      query = input("question:").strip()
      if query:
          result = querying_index(query,vectorstore)
          if result:
              print("\nAnswer:")
              print(result["answer"])

if __name__ == "__main__":
    main()
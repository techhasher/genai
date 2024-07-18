import boto3
import streamlit as st
import json

from langchain_community.document_loaders import S3FileLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
llm_model_id = "amazon.titan-text-express-v1"
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def data_ingestion():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('techhashirabucket')

    #Due to permission error seen on Windows while loading docx, doing it the hard way by downloading S3 files
    #and then using Docx2txtLoader.  ELse could use S3FileLoader directly to connect to S3 and load the files using loader.load 
    # as seen on https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.s3_file.S3FileLoader.html
    s3KeyList = []
    for obj in bucket.objects.all():
        s3KeyList.append(obj.key)
    #print(*s3KeyList, sep = "\n")

    #Uncomment when running for first time as you need to first download files.  Set your folder name. Not needed if using S3FileLoader directly
    #for s3_key in s3KeyList:
        #bucket.download_file(s3_key, f"C:\\Seena\\LLM\\{s3_key}")
    docs = []
    for s3_key in s3KeyList:
        with open(f"C:\\Seena\\LLM\\{s3_key}", "rb") as f:
            if "docx" in s3_key.lower():
                #loader = UnstructuredWordDocumentLoader(f"C:\\Seena\\LLM\\{s3_key}")
                loader = Docx2txtLoader(f"C:\\Seena\\LLM\\{s3_key}")
                print(f"Loaded {s3_key}")
                docs.extend(loader.load())
            elif "pdf" in s3_key.lower():
                loader = PyPDFDirectoryLoader(f"C:\\Seena\\LLM\\{s3_key}")
                print(f"Loaded {s3_key}")
                docs.extend(loader.load())
    print(docs[6])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=300)

    split_docs = text_splitter.split_documents(docs)
    return split_docs

## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings,
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llm():
    ##create the Anthropic Model
    llm = Bedrock(model_id=llm_model_id, client=bedrock,
                  model_kwargs={'temperature': 0})

    return llm


def get_response_llm(llm, vectorstore_faiss, query):
    prompt_template = """

    Human: Use the following pieces of context and draft an response 
    to the email at the end.  The response should be brief and to the point and
     should not exceed 100 words.  Provide all possible options. If the response has a list, present the list in a tabular format.
    Start the reply with Hi Abc and sign off with Regards.  Provide the reference documents and urls 
    if available from where you have sourced your answer. If you don't know the answer, 
    just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context

    Question: {question}

    Reply:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def summarize(email):
    prompt = f"""Human: As a middle office assistant,
    your job is to summarize the contents of the email.  You should use a professional tone.
    The details of the feedback is between the <data> XML like tags.
     <data>
    {email}
    </data>
    """
    
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "temperature": 0,
            "topP": 0.9,
            "maxTokenCount": 8000
        }
    })
    accept = "application/json"
    content_type = "application/json"
    print(body)
    response = bedrock.invoke_model(body=body, modelId=llm_model_id,accept=accept, contentType=content_type)
    response_body = json.loads(response.get('body').read())

    for result in response_body['results']:
        print(f"Token count: {result['tokenCount']}")
        print(f"Output text: {result['outputText']}")
        print(f"Completion reason: {result['completionReason']}")
        return {result['outputText']}




def main():
    # data_ingestion()
    st.set_page_config("Generate Email response")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    with st.expander("Generate Email response"):
        st.header("Generate email response using AWS Bedrock💁")
        user_question = st.text_input("Ask a Question")
        if st.button("Generate Response"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llm()

                # faiss_index = get_vector_store(docs)
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")

    with st.expander("Summarize Email"):
        st.header("Summarize Email:")
        email_content = st.text_input("Copy Email")
        if st.button("Summarize"):
            with st.spinner("Processing..."):
                print(email_content)
                st.write(summarize(email_content))
                st.success("Done")


if __name__ == "__main__":
    main()


# s3_resource = boto3.resource("s3")
# print("Hello, Amazon S3! Let's list your buckets:")
# for bucket in s3_resource.buckets.all():
#     print(f"\t{bucket.name}")
#
# loader = S3FileLoader("techhashirabucket", "Booking Rules for Entity1JP.docx")
# data = loader.load()
# print('{data}')

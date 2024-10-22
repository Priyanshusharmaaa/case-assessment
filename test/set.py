from typing import Literal
from dataclasses import dataclass
from langchain_groq import ChatGroq
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
import hnswlib
from annoy import AnnoyIndex
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["👤 Human", "👨🏻‍⚖️ Ai"]
    message: str

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# Initialize the conversation system
def initialize_conversation():
    llama = LlamaAPI("d9f5e52518c6432f9c568a308d17554b767f2543a040477d9888bf9704d4d065")
    model = ChatLlamaAPI(client=llama)
    chat = ChatGroq(
        temperature=0.5,
        groq_api_key="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",
        model_name="llama3-70b-8192"  # llama3-70b-8192
    )

    embeddings = download_hugging_face_embeddings()

    persist_directory = "./chroma-db"
    
    # Initialize different vector stores: FAISS, Annoy, HNSW
    chroma_client = Chroma(
        collection_name="legal-advisor",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    faiss_client = FAISS.from_documents([], embeddings)
    annoy_client = AnnoyIndex(collection_name="legal-advisor", embedding_function=embeddings)
    hnsw_client = HNSWLib.from_documents([], embeddings)

    prompt_template = """
        You are an Indian legal expert helping users generate marital settlement agreements. Based on the data in the vector database and the values given by the user and their religion, decide which marriage act to consider and fill in the values for:
        - Amount for Basic Payment Provisions
        - Amount under Basic Payment Provision (Spousal Support)
        - Termination of Jurisdiction (Husband/Wife)
        - Amount Payment to Balance Division
        - Amount for Maintenance Repairs
        - Percentage for Husband and Wife Tax Refunds under Allocation of Income Tax
        - Split of assets from total assets
        - Mention the marriage act and the reason behind it
        
        Return only the values in JSON format.
        Context: {context}
        Question: {question}
        Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create ConversationalRetrievalChain with each algorithm
    retrieval_chain_chroma = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=chroma_client.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        memory=memory
    )

    retrieval_chain_faiss = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=faiss_client.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        memory=memory
    )

    retrieval_chain_annoy = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=annoy_client.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        memory=memory
    )

    retrieval_chain_hnsw = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=hnsw_client.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        memory=memory
    )

    return {
        "chroma": retrieval_chain_chroma,
        "faiss": retrieval_chain_faiss,
        "annoy": retrieval_chain_annoy,
        "hnsw": retrieval_chain_hnsw
    }

def generate_document(conversation, user_inputs, algo_name):
    # Prepare the financial context for LLM
    financial_context = f"""
    Husband's Income: {user_inputs['Husband_Income']}
    Wife's Income: {user_inputs['Wife_Income']}
    Husband's Religion:{user_inputs['Husband_Religion']}
    Wife's Religion:{user_inputs['Wife_Religion']}
    Total Assets: {user_inputs['total_assets']}
    """

    # Combine the financial context and question into one prompt
    prompt = f"""
    Context: {financial_context}
    Fill in the following values for the marital settlement agreement:
    - Amount for Basic Payment Provisions
    - Amount under Basic Payment Provision (Spousal Support)
    - Husband/Wife Termination of Jurisdiction
    - Amount Payment to Balance Division
    - Amount for Maintenance Repairs
    - Percentage for Husband and Wife Tax Refunds under Allocation of Income Tax
    - Assets allocated to husband
    - Assets allocated to wife
    - Marriage Act
    """

    # Call the chain and get the answer
    response = conversation[algo_name]({"question": prompt})
    generated_values = response['answer']  # Extract only the answer

    # Display the generated values
    print(f"Output for {algo_name.upper()} algorithm:")
    print(generated_values)
    print()

# Example default inputs for testing
def get_default_user_details():
    return {
        "Husband_Name": "John Doe",
        "Wife_Name": "Jane Doe",
        "Husband_Religion": "Christian",
        "Wife_Religion": "Christian",
        "Marriage_Date": "2010-05-15",
        "Separation_Date": "2022-08-01",
        "Number_of_Children": 2,
        "Children": [
            {"name": "Alice Doe", "dob": "2012-03-20"},
            {"name": "Bob Doe", "dob": "2015-06-10"}
        ],
        "Husband_Income": 387332.00,
        "Wife_Income": 187040.00,
        "total_assets": [
            {"name": "House", "value": 500000.00},
            {"name": "Car", "value": 30000.00},
            {"name": "Stocks", "value": 150000.00}
        ]
    }

if __name__ == "__main__":
    conversation = initialize_conversation()
    user_inputs = get_default_user_details()
    
    # Test all algorithms
    for algo in ["chroma", "faiss", "annoy", "hnsw"]:
        generate_document(conversation, user_inputs, algo)
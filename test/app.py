import streamlit as st
from typing import Literal
from dataclasses import dataclass
from langchain_groq import ChatGroq
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
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

def initialize_conversation():
    llama = LlamaAPI("d9f5e52518c6432f9c568a308d17554b767f2543a040477d9888bf9704d4d065")
    model = ChatLlamaAPI(client=llama)
    chat = ChatGroq(
        temperature=0.5,
        groq_api_key="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",
        model_name="llama3-70b-8192"
    )

    embeddings = download_hugging_face_embeddings()
    persist_directory = "./chroma-db"
    chroma_client = Chroma(
        collection_name="legal-advisor",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    prompt_template = """
        You are a copyright case analyzing assistant. You will help the user with the copyright case by providing a fair assessment.
        Analyze the arguments of both sides, their supporting evidence, determine which argument is more convincing, and identify key points for both sides.
        Highlight any errors in their arguments and suggest improvements for the judgment.
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

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=chroma_client.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        memory=memory
    )

    return retrieval_chain

def generate_analysis(conversation, case_details):
    case_context = f"""
    Plaintiff's Claim: {case_details['plaintiff_claim']}
    Plaintiff's Evidence: {case_details['plaintiff_evidence']}
    
    Defendant's Argument: {case_details['defendant_argument']}
    Defendant's Evidence: {case_details['defendant_evidence']}
    
    Legal Context: {case_details['legal_context']}
    """

    prompt = f"""
    Context: {case_context}
    Analyze the case and provide:
    - Key points for both the plaintiff and the defendant
    - Analysis of the evidence supporting each argument
    - Which argument is stronger
    - Any errors or misinterpretations in their arguments
    - Suggestions for judgment improvement and fairness
    """

    response = conversation({"question": prompt})
    generated_analysis = response['answer']
    
    return generated_analysis

# Streamlit app
def main():
    st.title("Copyright Case Analysis Assistant")

    # Input fields for the user
    plaintiff_claim = st.text_area("Plaintiff's Claim", placeholder="Enter the plaintiff's claim here...")
    plaintiff_evidence = st.text_area("Plaintiff's Evidence", placeholder="Enter the plaintiff's evidence here...")

    defendant_argument = st.text_area("Defendant's Argument", placeholder="Enter the defendant's argument here...")
    defendant_evidence = st.text_area("Defendant's Evidence", placeholder="Enter the defendant's evidence here...")

    legal_context = st.text_area("Legal Context", placeholder="Enter the legal context here...")

    # Button to generate analysis
    if st.button("Analyze Case"):
        if not plaintiff_claim or not plaintiff_evidence or not defendant_argument or not defendant_evidence or not legal_context:
            st.warning("Please fill in all fields.")
        else:
            case_details = {
                "plaintiff_claim": plaintiff_claim,
                "plaintiff_evidence": plaintiff_evidence,
                "defendant_argument": defendant_argument,
                "defendant_evidence": defendant_evidence,
                "legal_context": legal_context
            }
            conversation = initialize_conversation()
            analysis = generate_analysis(conversation, case_details)

            # Display the analysis
            st.subheader("Generated Case Analysis")
            st.write(analysis)

if __name__ == "__main__":
    main()
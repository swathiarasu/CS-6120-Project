# import streamlit as st
# import fitz  # PyMuPDF
# from components.retriever import Retriever
# from components.rag_model import RAGModel

# def extract_text_from_pdf(uploaded_file):
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# def chunk_text(text, chunk_size=500, overlap=50):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = words[i:i + chunk_size]
#         chunks.append(" ".join(chunk))
#     return chunks

# st.title("arXiv Research Paper Q&A with Citations")

# uploaded_file = st.file_uploader("Upload an arXiv PDF paper", type="pdf")

# if uploaded_file:
#     st.success("PDF uploaded. Extracting text...")
#     text = extract_text_from_pdf(uploaded_file)
#     chunks = chunk_text(text)
#     retriever = Retriever(chunks)
#     rag = RAGModel()

#     question = st.text_input("Ask a question about the paper:")

#     if question:
#         with st.spinner("Retrieving context and generating answer..."):
#             top_chunks = retriever.get_top_k(question, k=3)
#             context = "\n".join(top_chunks)
#             answer = rag.generate_answer(question, context)

#             citations = " ".join([f"[{i+1}]" for i in range(len(top_chunks))])
#             answer_with_citations = answer + " " + citations

#             st.subheader("Answer:")
#             st.write(answer_with_citations)

#             with st.expander("Cited Passages"):
#                 for i, chunk in enumerate(top_chunks):
#                     st.markdown(f"**[{i+1}]** {chunk}")


#######################################################################################################

# import streamlit as st
# from components.retriever_multi import MultiPDFRetriever
# from components.rag_model import RAGModel

# st.title("Ask Questions Across Multiple arXiv Papers")

# retriever = MultiPDFRetriever()
# rag = RAGModel()

# question = st.text_input("Ask a question across the dataset:")

# if question:
#     with st.spinner("Retrieving answers from indexed arXiv papers..."):
#         top_chunks = retriever.get_top_k(question, k=3)
#         context = "\n".join([c[2] for c in top_chunks])
#         answer = rag.generate_answer(question, context)

#         st.subheader("Answer:")
#         st.write(answer + " " + " ".join([f"[{i+1}]" for i in range(len(top_chunks))]))

#         with st.expander("Cited Sources"):
#             for i, (filename, chunk_id, chunk_text) in enumerate(top_chunks):
#                 st.markdown(f"**[{i+1}] {filename}** (chunk {chunk_id})\n\n{chunk_text}")


#######################################################################################################


# import streamlit as st
# from components.rag_model import RAGModel
# from components.retriever_multi import MultiPDFRetriever 
# from components.retriever_live import LiveRetriever 
# import torch
# import fitz 


# torch.backends.mps.is_available = lambda: False
# device = torch.device("cpu")

# # Streamlit App
# def main():
#     st.set_page_config(page_title="arXiv RAG - Ask a Research Question")
#     st.title("Ask Questions Across Your arXiv Dataset")

#     # Load FAISS index + metadata
#     with st.spinner("🔍 Loading RAG components..."):
#         retriever = MultiPDFRetriever(
#             index_path="data/index/faiss_index.bin",
#             metadata_path="data/index/metadata.pkl"
#         )
#         rag = RAGModel(device=device)

#     # Input box
#     question = st.text_input("Ask a question:")

#     if question:
#         with st.spinner("🧠 Thinking..."):
#             top_chunks = retriever.get_top_k(question, k=3)
#             context = "\n".join([c[2] for c in top_chunks])
#             answer = rag.generate_answer(question, context)

#             st.subheader("Answer")
#             st.markdown(answer + " " + " ".join([f"[{i+1}]" for i in range(len(top_chunks))]))

#             with st.expander("📚 Cited Sources"):
#                 for i, (title, chunk_id, chunk_text) in enumerate(top_chunks):
#                     st.markdown(f"**[{i+1}] {title}** (chunk {chunk_id})\n\n{chunk_text}")

# if __name__ == "__main__":
#     main()


import streamlit as st
import fitz  # For PDF reading
import torch
from components.rag_model import RAGModel
from components.retriever_multi import MultiPDFRetriever  # For FAISS-based search
from components.retriever_live import LiveRetriever      # For uploaded document search

# Force CPU to avoid MacOS GPU issues (MPS)
torch.backends.mps.is_available = lambda: False
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Extract text from uploaded file (PDF or .txt)
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return ""

# Main Streamlit App
def main():
    st.set_page_config(page_title="Hybrid RAG QA System")
    st.title("AcademEase: RAG Model for Research Assistance")

    # Load retrievers and model
    with st.spinner("Loading index and model..."):
        faiss_retriever = MultiPDFRetriever(
            index_path="data/index/faiss_index.bin",
            metadata_path="data/index/metadata.pkl"
        )
        live_retriever = LiveRetriever()
        rag = RAGModel()

    # File uploader
    uploaded_file = st.file_uploader("Optionally upload a PDF or .txt file", type=["pdf", "txt"])

    # User query input
    question = st.text_input("Ask a question about your dataset and uploaded file:")

    if question:
        with st.spinner("Retrieving and generating answer..."):
            # Search FAISS index
            faiss_results = faiss_retriever.get_top_k(question, k=3)

            # Search uploaded file
            upload_context = ""
            if uploaded_file:
                doc_text = extract_text(uploaded_file)
                live_results = live_retriever.get_top_k_from_text(doc_text, question, k=3)
                upload_context = "\n".join([x[1] for x in live_results])
            else:
                live_results = []

            # Merge all context
            faiss_context = "\n".join([chunk[2] for chunk in faiss_results])
            full_context = faiss_context + "\n" + upload_context

            # Generate answer
            answer = rag.generate_answer(question, full_context)

            # Display output
            st.subheader("Answer")
            st.markdown(answer + " " + " ".join([f"[{i+1}]" for i in range(len(faiss_results + live_results))]))

            with st.expander("📚 Sources from FAISS index"):
                for i, (title, chunk_id, chunk_text) in enumerate(faiss_results):
                    st.markdown(f"**[{i+1}] {title}** (chunk {chunk_id})\n\n{chunk_text}")

            if uploaded_file:
                with st.expander("📄 Sources from Uploaded File"):
                    for i, (chunk_id, chunk_text) in enumerate(live_results, start=len(faiss_results)+1):
                        st.markdown(f"**[{i}] Uploaded Doc (chunk {chunk_id})**\n\n{chunk_text}")

if __name__ == "__main__":
    main()

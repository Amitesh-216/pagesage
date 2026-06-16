
import os
import base64
import tempfile
from io import BytesIO
from dotenv import load_dotenv
 
import streamlit as st
import fitz  
from PIL import Image
import pytesseract
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
 
load_dotenv()
 
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )
 
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    )
 
@st.cache_resource
def get_vision_llm():
    # meta-llama/llama-4-scout-17b-16e-instruct supports vision on Groq
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    )
 
def image_to_base64(pil_image: Image.Image) -> str:
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
 
def describe_image_with_vision(pil_image: Image.Image, page_num: int) -> str:
    """Send image to Groq vision model to describe diagrams, circuits, phasors."""
    try:
        from groq import Groq
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
        )
        b64 = image_to_base64(pil_image)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "This image is from a technical/educational PDF. "
                                "Describe everything you see in detail: any circuit diagrams, "
                                "phasor diagrams, graphs, equations, tables, or text. "
                                "Be thorough and technical so this description can be used "
                                "to answer questions about the diagram."
                            )
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Image on page {page_num+1} could not be described: {e}]"
 
def load_and_chunk(pdf_path: str):
    """
    Extract text + images from PDF using PyMuPDF.
    - Text pages: extracted directly
    - Scanned/image pages: OCR via pytesseract
    - Diagrams/figures: described via Groq vision model
    """
    doc = fitz.open(pdf_path)
    all_documents = []
 
    progress = st.progress(0, text="Extracting PDF content...")
 
    for page_num, page in enumerate(doc):
        progress.progress(
            (page_num + 1) / len(doc),
            text=f"Processing page {page_num + 1} of {len(doc)}..."
        )
 
       ──────────────────────────────────────────
        text = page.get_text().strip()
 
        if text and len(text) > 30:
            
            all_documents.append(Document(
                page_content=text,
                metadata={"page": page_num, "type": "text"}
            ))
        else:
            
            mat = fitz.Matrix(2.0, 2.0)  
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
 
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text and len(ocr_text) > 20:
                all_documents.append(Document(
                    page_content=f"[OCR - Page {page_num+1}]\n{ocr_text}",
                    metadata={"page": page_num, "type": "ocr"}
                ))
 
       ────────────────────
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
 
                if pil_image.width > 100 and pil_image.height > 100:
                    description = describe_image_with_vision(pil_image, page_num)
                    all_documents.append(Document(
                        page_content=f"[Diagram/Figure on Page {page_num+1}]\n{description}",
                        metadata={"page": page_num, "type": "image"}
                    ))
            except Exception:
                continue
 
    progress.empty()
    doc.close()
 
    if not all_documents:
        raise ValueError("No content could be extracted from this PDF.")
 
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(all_documents)
    chunks = [c for c in chunks if c.page_content.strip()]
 
    if not chunks:
        raise ValueError("PDF contains no readable content.")
 
    return chunks
 
def build_vector_store(chunks):
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)
 
def build_chain(vector_store):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
 
def ask(retriever, question: str, chat_history: list):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = sorted(set(doc.metadata.get("page", 0) + 1 for doc in docs))
 
    history_text = "\n".join(
        f"Human: {q}\nAssistant: {a}" for q, a, _ in chat_history
    )
 
    prompt = PromptTemplate.from_template("""
You are a helpful technical assistant. Use the context below to answer the question.
The context may include text, OCR-extracted content, and descriptions of diagrams,
circuit diagrams, and phasor diagrams from a PDF.
 
CRITICAL FORMATTING RULES — always follow these:
1. For numerical problems, show the full step-by-step solution clearly.
2. Write all formulas and equations using proper notation:
   - Use ** for powers (e.g. V**2 / R or write as fraction)
   - Use × for multiplication
   - Write fractions as: numerator / denominator on separate lines with a dividing line using markdown
   - Use bullet points or numbered steps for each calculation step
3. Never write raw LaTeX code like fraction or sqrt() — write it in plain readable form.
4. Use markdown bold (**text**) for final answers.
5. Use tables where comparisons or multiple values are involved.
6. If the answer involves a diagram or circuit, explain it clearly and in detail.
7. If the answer is not in the context, say "I don't know based on this document."
 
Example of good numerical formatting:
**Given:** V = 230 V, R = 10 Ω
 
**Step 1 — Find Current:**
I = V / R = 230 / 10 = **23 A**
 
**Step 2 — Find Power:**
P = V × I = 230 × 23 = **5290 W**
 
**Final Answer: P = 5290 W**
 
Context:
{context}
 
Chat History:
{chat_history}
 
Question: {question}
 
Answer:""")
 
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
 
    answer = chain.invoke({
        "context": context,
        "chat_history": history_text,
        "question": question
    })
 
    return answer, sources
 

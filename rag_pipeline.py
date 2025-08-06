import os
import json
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TRANSCRIPTS_DIR = "transcripts"
VECTORSTORE_DIR = "vectorstores"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# ✅ Caching transcript
def get_transcript_text(video_id: str) -> str:
    cache_file = f"{TRANSCRIPTS_DIR}/{video_id}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)["text"]

    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        text = " ".join([snippet.text for snippet in fetched_transcript])

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"text": text}, f)

        return text
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        return "No transcript available for this video."
    except Exception as e:
        return "An unexpected error occurred while fetching the transcript."


# ✅ Caching FAISS vectorstore
def get_or_create_vectorstore(video_id: str, chunks):
    index_path = os.path.join(VECTORSTORE_DIR, video_id)
    if os.path.exists(index_path):
        return FAISS.load_local(
            folder_path=index_path,
            embeddings=GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY
            ),
            allow_dangerous_deserialization=True
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore


# ✅ Main RAG pipeline
def get_rag_answer(question: str, video_id: str) -> str:
    transcript_text = get_transcript_text(video_id)
    if not transcript_text or "No transcript" in transcript_text:
        return transcript_text

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    vectorstore = get_or_create_vectorstore(video_id, chunks)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer only from the provided transcript context.
        If the context is insufficient, just say you don't know.
        {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return " ".join([doc.page_content for doc in docs])

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough()
        ) | prompt | llm | StrOutputParser()
    )

    return chain.invoke(question)

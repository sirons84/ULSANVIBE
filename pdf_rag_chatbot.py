import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import os

# 페이지 설정
st.set_page_config(
    page_title="PDF 챗봇",
    page_icon="📚",
    layout="wide"
)

# CSS 스타일링
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

# 타이틀
st.title("📚 PDF 기반 AI 챗봇")
st.markdown("**test.pdf** 문서에 대해 무엇이든 물어보세요!")

# API 키 확인
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ API 키가 설정되지 않았습니다. Streamlit secrets에 GEMINI_API_KEY를 추가해주세요.")
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# PDF 처리 함수
@st.cache_resource
def process_pdf():
    """PDF를 로드하고 벡터 스토어를 생성"""
    try:
        # PDF 읽기
        pdf_reader = PdfReader("test.pdf")
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # 임베딩 및 벡터 스토어 생성
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        return vectorstore
    except FileNotFoundError:
        st.error("⚠️ test.pdf 파일을 찾을 수 없습니다. 파일이 프로젝트 루트에 있는지 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"⚠️ PDF 처리 중 오류가 발생했습니다: {str(e)}")
        return None

# RAG 체인 생성
def create_chain(vectorstore):
    """Conversational Retrieval Chain 생성"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    return chain

# 사이드바
with st.sidebar:
    st.header("ℹ️ 정보")
    st.markdown("""
    이 챗봇은 **test.pdf** 문서의 내용을 기반으로 답변합니다.
    
    **기능:**
    - 📄 PDF 문서 분석
    - 🤖 Gemini 2.5 Flash 모델
    - 🔍 RAG 기반 정확한 답변
    - 💬 대화 기록 유지
    """)
    
    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.processed = False
        st.rerun()

# PDF 처리 (최초 1회)
if not st.session_state.processed:
    with st.spinner("📄 PDF 문서를 분석하는 중..."):
        vectorstore = process_pdf()
        if vectorstore:
            st.session_state.chain = create_chain(vectorstore)
            st.session_state.processed = True
            st.success("✅ 문서 분석 완료! 질문을 입력해주세요.")
        else:
            st.stop()

# 채팅 기록 표시
for message in st.session_state.messages:
    css_class = "user-message" if message["role"] == "user" else "assistant-message"
    icon = "👤" if message["role"] == "user" else "🤖"
    
    st.markdown(f"""
        <div class="chat-message {css_class}">
            <div style="font-weight: bold; margin-bottom: 0.5rem;">{icon} {message["role"].upper()}</div>
            <div>{message["content"]}</div>
        </div>
    """, unsafe_allow_html=True)

# 채팅 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 사용자 메시지 표시
    st.markdown(f"""
        <div class="chat-message user-message">
            <div style="font-weight: bold; margin-bottom: 0.5rem;">👤 USER</div>
            <div>{prompt}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # AI 응답 생성
    with st.spinner("🤔 답변 생성 중..."):
        try:
            response = st.session_state.chain({"question": prompt})
            answer = response["answer"]
            
            # AI 메시지 추가
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # AI 메시지 표시
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">🤖 ASSISTANT</div>
                    <div>{answer}</div>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"⚠️ 오류가 발생했습니다: {str(e)}")

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Powered by Gemini 2.5 Flash & LangChain</div>",
    unsafe_allow_html=True
)
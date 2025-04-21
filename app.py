import streamlit as st
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente (.env com OPENAI_API_KEY)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Configurações iniciais do app
st.set_page_config(page_title="Assistente Contratual Flash", page_icon="📄")
st.title("🤖 Mingo - Assistente Contratual Flash")
st.markdown("Pergunte sobre cláusulas do contrato. O Mingo responde com base nos comentários jurídicos da Flash!")

# Carrega o índice (usa cache pra não recarregar sempre)
@st.cache_resource
def carregar_assistente():
    # Lê os arquivos da pasta clausulas/
    documents = SimpleDirectoryReader(input_dir="clausulas").load_data()

    # Define embeddings e LLM
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = OpenAI(model="llama3-8b-8192", api_key=api_key, temperature=0.1)

    # Cria o índice vetorial
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index.as_query_engine()

query_engine = carregar_assistente()

# Controle do histórico de conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input do usuário
user_input = st.text_input("Digite sua pergunta:", key="input")

if user_input:
    resposta = query_engine.query(user_input)
    st.session_state.chat_history.append(("Você", user_input))
    st.session_state.chat_history.append(("Mingo", str(resposta)))

# Mostrar histórico em estilo de chat
for autor, mensagem in st.session_state.chat_history:
    with st.chat_message("user" if autor == "Você" else "assistant"):
        st.markdown(mensagem)

# Botão para limpar conversa
if st.button("🧹 Limpar conversa"):
    st.session_state.chat_history = []

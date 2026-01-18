import streamlit as st
from rag_chat import RagChat

st.title("Chat with your pdf file")

uploaded_pdfs = st.sidebar.file_uploader("Choose your files", accept_multiple_files=True, type="pdf")

if 'uploaded_pdfs' not in st.session_state:
  st.session_state['uploaded_pdfs'] = []

if 'rag_chat' not in st.session_state:
  st.session_state['rag_chat'] = RagChat()

if 'chat_messsages' not in st.session_state:
  st.session_state['chat_messsages'] = []

session_uploaded_files = st.session_state['uploaded_pdfs']
current_uploaded_files = []

with st.sidebar:
  if len(uploaded_pdfs) > 0:
    with st.status("Handling the uploaded files...", expanded=True) as status:
      for uploaded_pdf in uploaded_pdfs:
        current_uploaded_files.append(uploaded_pdf.name)
        st.write(f"Handling {uploaded_pdf.name}...")
        if uploaded_pdf.name in session_uploaded_files:
          continue
        st.session_state['rag_chat'].handle_pdf(uploaded_pdf, uploaded_pdf.name)
      st.session_state['uploaded_pdfs'] = current_uploaded_files

      status.update(label="Uploaded files handled!", state="complete", expanded=False)

query = st.chat_input("Say something")

if query:
  st.session_state['chat_messsages'].append({
    'owner': 'user',
    'content': query,
  })
  
  response = st.session_state['rag_chat'].answer_user_questions(query)
  st.session_state['chat_messsages'].append({
    'owner': 'assistant',
    'content': response,
  })

for chat_msg in st.session_state['chat_messsages']:
  st.chat_message(chat_msg['owner']).write(chat_msg['content'])

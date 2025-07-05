# pip install faiss-cpu --no-cache
# pip install -q -U google-generativeai
# pip install python-dotenv
# pip install streamlit

import streamlit as st
import RagLlm as rag

ocr_system = rag.OCR_RAG_System(knowledge_base_path='knowledge_base.pkl')
#image_path = upload_file()
#response = ocr_system.process_image_with_rag(image_path, True)

st.title("Arabic Handwritten Text OCR Service")
st.header("Upload an image file")
uploaded_file = st.file_uploader("Select an image file (.jpg)", type=["jpg"])
image_path = None
response = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        image_path = temp_file.name
        response = ocr_system.process_image_with_rag(image_path, True)
        ocr_system.save_knowledge_base()

col1, col2 = st.columns(2)
with col1:
  if(image_path is not None and os.path.exists(image_path)):
    st.image(f"{image_path}_.jpg")
with col2:
  if(response is not None):
    st.text(response)

if(response is not None):
  if(st.button("Read the text for me!")):
    rag.ReadText(response)

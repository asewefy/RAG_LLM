from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
from markupsafe import escape
import RagLlm

app = Flask(__name__)

@app.route("/api/ragllm", methods=['POST'])
def Rag():
    #return f"Hello, {escape(name)}!"
    if request.method == 'POST':
        f = request.files['image']
        image_path = f'/var/www/uploads/{secure_filename(f.filename)}'
        f.save(image_path)
        ocr_system = rag.OCR_RAG_System(knowledge_base_path='knowledge_base.pkl')
        response = ocr_system.process_image_with_rag(image_path, True)
        ocr_system.save_knowledge_base()
        return response

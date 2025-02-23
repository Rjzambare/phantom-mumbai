from flask import Flask, request, jsonify, send_from_directory
from pdf_processor import extract_text_pypdf2
from rag_pipeline import setup_vector_store, query_rag
from chat_manager import ChatSession
import os

app = Flask(__name__)
sessions = {}  # Temporary in-memory storage for chat sessions

@app.route('/')
def home():
    return jsonify({"status": "OK", "message": "Chat API Service Running"})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Endpoint for patient to upload a PDF report
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files or 'session_id' not in request.form:
        return jsonify({"error": "Missing file or session_id"}), 400
        
    file = request.files['file']
    session_id = request.form['session_id']
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    file_path = f"temp_{session_id}.pdf"
    
    try:
        file.save(file_path)
        text = extract_text_pypdf2(file_path)
        vector_store = setup_vector_store(text)
        
        if session_id in sessions:
            sessions[session_id].vector_store = vector_store
        else:
            sessions[session_id] = ChatSession(session_id, vector_store)
            
        os.remove(file_path)
        
        return jsonify({
            "message": "PDF uploaded and processed",
            "session_id": session_id,
            "text_length": len(text)
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

# Endpoint for real-time chat between patient and AI avatar
@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    session_id = data.get('session_id')
    query = data.get('query')
    
    if not session_id or not query:
        return jsonify({"error": "Missing session_id or query"}), 400
    
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    try:
        session = sessions[session_id]
        response = session.handle_query(query)
        return jsonify({
            "response": response,
            "history": session.history[-5:]  # Return last 5 messages for context
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to end the chat session and generate the report
@app.route('/end_chat', methods=['POST'])
def end_chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    try:
        session = sessions[session_id]
        summary = session.generate_summary()
        report_file_path = session.generate_report()
        duration = session.get_duration()
        # Remove session from memory after generating the report
        del sessions[session_id]
        return jsonify({
            "summary": summary,
            "report_file": report_file_path,
            "session_duration": duration
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to download the generated PDF report
@app.route('/download_report', methods=['POST'])
def download_report():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    report_file = f"patient_report_{session_id}.pdf"
    if not os.path.exists(report_file):
        return jsonify({"error": "Report not found"}), 404

    try:
        return send_from_directory(directory=os.getcwd(), path=report_file, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

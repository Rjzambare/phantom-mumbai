from flask import Flask, request, jsonify
from pdf_processor import extract_text_pypdf2
from rag_pipeline import setup_vector_store, query_rag
from chat_manager import ChatSession
import os

app = Flask(__name__)
sessions = {}  # Temporary in-memory storage for chat sessions

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    session_id = request.form['session_id']
    
    # Save file temporarily and extract text
    file_path = f"temp_{session_id}.pdf"
    file.save(file_path)
    text = extract_text_from_pdf(file_path)
    
    # Setup vector store for this session
    vector_store = setup_vector_store(text)
    
    # Initialize chat session
    sessions[session_id] = ChatSession(session_id, vector_store)
    os.remove(file_path)  # Clean up
    return jsonify({"message": "PDF uploaded and processed", "session_id": session_id})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data['session_id']
    query = data['query']
    
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    response = session.handle_query(query)
    return jsonify({"response": response, "history": session.history})

@app.route('/end_chat', methods=['POST'])
def end_chat():
    data = request.json
    session_id = data['session_id']
    
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    summary = session.generate_summary()
    del sessions[session_id]  # Clean up session
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

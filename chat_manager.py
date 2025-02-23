from groq import Groq
from rag_pipeline import query_rag
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
import os
import time

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatSession:
    def __init__(self, session_id, vector_store):
        self.session_id = session_id
        self.vector_store = vector_store
        self.history = []
        self.start_time = time.time()
    
    def handle_query(self, query):
        response = query_rag(self.vector_store, query)
        self.history.append({"query": query, "response": response})
        return response
    
    def generate_summary(self):
        history_text = ""
        for entry in self.history:
            history_text += f"Q: {entry['query']}\nA: {entry['response']}\n\n"
        
        system_prompt = (
            "You are a medical assistant. Summarize the following patient chat history "
            "and any uploaded report information into a concise summary that can be used "
            "to resume the chat later and for record keeping."
        )
        full_prompt = f"Chat History:\n{history_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=512,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        summary = ""
        for chunk in completion:
            summary += chunk.choices[0].delta.content or ""
        
        return summary

    def generate_report(self):
        summary = self.generate_summary()
        file_path = f"patient_report_{self.session_id}.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=letter)

        table_data = [
            ["Session ID", self.session_id],
            ["Summary", summary],
            ["Chat History", ""]
        ]
        for entry in self.history:
            table_data.append(["Query", entry["query"]])
            table_data.append(["Response", entry["response"]])
            table_data.append(["", ""])  # Blank row for spacing

        table = Table(table_data, colWidths=[150, 400])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))

        doc.build([table])
        return file_path

    def get_duration(self):
        return int(time.time() - self.start_time)

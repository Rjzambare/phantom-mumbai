from datetime import datetime
from groq import Groq
from rag_pipeline import query_rag
import os

# Initialize Groq client (same as in rag_pipeline.py)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatSession:
    def __init__(self, session_id, vector_store):
        self.session_id = session_id
        self.vector_store = vector_store
        self.history = []
        # Initialize both timestamps
        self.start_time = datetime.now()
        self.end_time = None  # Will be set when session ends
    
    def handle_query(self, query):
        response = query_rag(self.vector_store, query)
        self.history.append({"query": query, "response": response})
        return response
    
    def generate_summary(self):
        # Set end_time when generating summary
        if not self.end_time:
            self.end_time = datetime.now()
            
        # Prepare the chat history as context
        history_text = ""
        for entry in self.history:
            history_text += f"Q: {entry['query']}\nA: {entry['response']}\n\n"
        
        # System prompt for summarization
        system_prompt = "You are a medical assistant. Summarize the following patient chat history into a concise and accurate summary."
        full_prompt = f"Chat History:\n{history_text}"

        # Prepare messages for Groq API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        # Call Groq API for summary
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,  # Lower temperature for more focused summary
            max_completion_tokens=512,  # Shorter output for summary
            top_p=1,
            stream=True,
            stop=None,
        )
        
        # Collect summary from streaming chunks
        summary = ""
        for chunk in completion:
            summary += chunk.choices[0].delta.content or ""
        
        return summary
        
    def get_duration(self):
        # Calculate duration safely
        end_time = self.end_time or datetime.now()
        duration = end_time - self.start_time
        return str(duration).split('.')[0]  # Remove microseconds

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import tempfile
import google.generativeai as genai
from supabase import create_client, Client
import fitz  # PyMuPDF
import pdfplumber
from pydantic import BaseModel
from typing import List, Optional
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
import aiofiles
import asyncio
from datetime import datetime
import requests

# Load environment variables
load_dotenv()

# Configuration
supabase_url = os.getenv("SUPABASE_URL")
supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize services
app = FastAPI(title="PDF Study Assistant API", version="1.0.0")
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase
supabase: Client = create_client(supabase_url, supabase_anon_key)

# Initialize Gemini AI
genai.configure(api_key=gemini_api_key)

# Firebase Admin initialization
try:
    firebase_admin.initialize_app(credentials.Certificate("firebase-service-account.json"))
    print("Firebase Admin initialized successfully")
except Exception as e:
    print(f"Firebase initialization error: {e}")

# In-memory storage for uploaded files
uploaded_files = {}

# Pydantic models
class PDFUploadResponse(BaseModel):
    status: str
    file_id: str
    filename: str
    file_url: str

class ProcessingResponse(BaseModel):
    status: str
    result: str

class SummarizeRequest(BaseModel):
    file_id: str
    summary_type: str = "brief"

class QuestionRequest(BaseModel):
    file_id: str
    question: str

class QuizRequest(BaseModel):
    file_id: str
    difficulty: str = "medium"
    num_questions: int = 5

class FlashcardsRequest(BaseModel):
    file_id: str
    num_cards: int = 10

# Helper functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using multiple methods"""
    text = ""
    
    try:
        # Try PyMuPDF first
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if text.strip():
            return text
    except Exception as e1:
        print(f"PyMuPDF failed: {e1}")
        
        try:
            # Fallback to pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e2}")

async def get_file_content(file_id: str) -> bytes:
    """Get file content from either in-memory storage or Supabase"""
    print(f"Getting file content for: {file_id}")
    
    if file_id in uploaded_files:
        print(f"File found in memory: {file_id}")
        file_info = uploaded_files[file_id]
        file_url = file_info['file_url']
        
        try:
            response = requests.get(file_url, timeout=30)
            if response.status_code == 200:
                print(f"Downloaded file from URL, size: {len(response.content)} bytes")
                return response.content
            else:
                print(f"Failed to download from URL: {response.status_code}")
                raise HTTPException(status_code=500, detail=f"Cannot download file from URL. Check Supabase bucket policies.")
        except requests.exceptions.RequestException as req_error:
            print(f"Request failed: {req_error}")
            raise HTTPException(status_code=500, detail=f"Network error downloading file: {req_error}")
    else:
        try:
            print(f"Downloading from Supabase storage: {file_id}")
            file_data = supabase.storage.from_("pdfs").download(file_id)
            print(f"Downloaded from Supabase, size: {len(file_data)} bytes")
            return file_data
        except Exception as download_error:
            print(f"Supabase download failed: {download_error}")
            raise HTTPException(status_code=500, detail=f"Supabase storage error: {download_error}")

async def process_with_gemini_latest(prompt: str, text: str) -> str:
    """Process text with latest Gemini 1.5 models"""
    try:
        print(f"Starting Gemini 1.5 Flash processing...")
        print(f"Text length: {len(text)} characters")
        
        # Use latest Gemini 1.5 Flash model (faster, more efficient)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Better text chunking for large documents
        if len(text) > 50000:  # 1.5 models can handle more
            chunks = [text[i:i+50000] for i in range(0, len(text), 45000)]  # 5k overlap
            chunk_results = []
            
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"{prompt}\n\nDocument Section {i+1}/{len(chunks)}:\n{chunk}"
                print(f"Processing chunk {i+1}/{len(chunks)}")
                
                response = model.generate_content(chunk_prompt)
                if response and response.text:
                    chunk_results.append(response.text)
            
            # Combine results intelligently
            if len(chunk_results) > 1:
                final_prompt = f"Synthesize these section analyses into a comprehensive {prompt.split(':')[0].lower()}:\n\n" + "\n\n---\n\n".join(chunk_results)
                final_response = model.generate_content(final_prompt)
                return final_response.text if final_response and final_response.text else "Failed to synthesize results"
            else:
                return chunk_results[0] if chunk_results else "No content processed"
        else:
            # Single request for shorter texts
            full_prompt = f"{prompt}\n\nDocument Content:\n{text}"
            response = model.generate_content(full_prompt)
            return response.text if response and response.text else "Empty response from AI"
        
    except Exception as e:
        print(f"Gemini 1.5 error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

# API Routes
@app.get("/")
async def root():
    return {"message": "PDF Study Assistant API", "version": "2.0.0", "model": "gemini-1.5-flash"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "supabase": bool(supabase),
            "gemini": bool(gemini_api_key),
            "firebase": bool(firebase_admin._apps)
        },
        "ai_model": "gemini-1.5-flash"
    }

@app.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    user = Depends(verify_token)
):
    """Upload PDF to Supabase storage"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        user_id = user['uid']
        file_path = f"user_uploads/{user_id}/{file.filename}"
        
        # Check if file already exists
        if file_path in uploaded_files:
            print(f"File already exists in memory: {file_path}")
            existing_file = uploaded_files[file_path]
            return PDFUploadResponse(
                status="exists",
                file_id=file_path,
                filename=file.filename,
                file_url=existing_file['file_url']
            )

        file_content = await file.read()
        
        # Upload to Supabase
        upload_response = supabase.storage.from_("pdfs").upload(
            file_path, file_content,
            file_options={"content-type": "application/pdf"}
        )
        
        # Get public URL
        public_url_response = supabase.storage.from_("pdfs").get_public_url(file_path)
        file_url = public_url_response['publicUrl']
        
        # Store in memory for quick access
        uploaded_files[file_path] = {
            'filename': file.filename,
            'file_url': file_url,
            'size': len(file_content),
            'uploaded_at': datetime.now().isoformat()
        }
        
        return PDFUploadResponse(
            status="success",
            file_id=file_path,
            filename=file.filename,
            file_url=file_url
        )
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.get("/documents")
async def get_documents(user = Depends(verify_token)):
    """Get list of user's uploaded documents"""
    try:
        user_id = user['uid']
        user_files = []
        
        for file_id, file_info in uploaded_files.items():
            if file_id.startswith(f"user_uploads/{user_id}/"):
                user_files.append({
                    "id": file_id,
                    "filename": file_info['filename'],
                    "file_url": file_info['file_url'],
                    "size": file_info['size'],
                    "uploaded_at": file_info['uploaded_at']
                })
        
        return {"documents": user_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {e}")

@app.post("/summarize", response_model=ProcessingResponse)
async def summarize_pdf(
    request: SummarizeRequest,
    user = Depends(verify_token)
):
    """Generate summary using Gemini 1.5 Flash"""
    try:
        print(f"Summarizing PDF: {request.file_id}")
        
        file_content = await get_file_content(request.file_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        text = extract_text_from_pdf(tmp_file_path)
        os.unlink(tmp_file_path)
        
        # Enhanced prompts for different summary types
        prompts = {
            "brief": "Create a concise 2-3 paragraph summary highlighting the main points and key takeaways:",
            "detailed": "Provide a comprehensive summary with detailed analysis, key points, main arguments, and important details:",
            "bullet_points": "Create a well-structured bullet-point summary with main topics, subtopics, and key details:"
        }
        
        prompt = prompts.get(request.summary_type, prompts["brief"])
        summary = await process_with_gemini_latest(prompt, text)
        
        return ProcessingResponse(status="success", result=summary)
        
    except Exception as e:
        print(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

@app.post("/ask", response_model=ProcessingResponse)
async def ask_question(
    request: QuestionRequest,
    user = Depends(verify_token)
):
    """Answer questions using Gemini 1.5 Flash"""
    try:
        print(f"Processing question about PDF: {request.file_id}")
        
        file_content = await get_file_content(request.file_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        text = extract_text_from_pdf(tmp_file_path)
        os.unlink(tmp_file_path)
        
        prompt = f"Based on the document content, provide a detailed and accurate answer to this question: {request.question}"
        answer = await process_with_gemini_latest(prompt, text)
        
        return ProcessingResponse(status="success", result=answer)
        
    except Exception as e:
        print(f"Question processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {e}")

@app.post("/quiz", response_model=ProcessingResponse)
async def generate_quiz(
    request: QuizRequest,
    user = Depends(verify_token)
):
    """Generate quiz using Gemini 1.5 Flash"""
    try:
        print(f"Generating quiz for PDF: {request.file_id}")
        
        file_content = await get_file_content(request.file_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        text = extract_text_from_pdf(tmp_file_path)
        os.unlink(tmp_file_path)
        
        prompt = f"""Create a {request.difficulty} difficulty quiz with {request.num_questions} multiple-choice questions based on the document content. 
        Format as JSON with this structure:
        {{
          "questions": [
            {{
              "question": "Question text",
              "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
              "correct_answer": "A",
              "explanation": "Why this answer is correct"
            }}
          ]
        }}"""
        
        quiz = await process_with_gemini_latest(prompt, text)
        
        return ProcessingResponse(status="success", result=quiz)
        
    except Exception as e:
        print(f"Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {e}")

@app.post("/flashcards", response_model=ProcessingResponse)
async def generate_flashcards(
    request: FlashcardsRequest,
    user = Depends(verify_token)
):
    """Generate flashcards using Gemini 1.5 Flash"""
    try:
        print(f"Generating flashcards for PDF: {request.file_id}")
        
        file_content = await get_file_content(request.file_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        text = extract_text_from_pdf(tmp_file_path)
        os.unlink(tmp_file_path)
        
        prompt = f"""Create {request.num_cards} educational flashcards based on the document content.
        Format as JSON with this structure:
        {{
          "flashcards": [
            {{
              "front": "Question or key term",
              "back": "Answer or definition",
              "category": "Topic category"
            }}
          ]
        }}
        Focus on key concepts, definitions, important facts, and main ideas."""
        
        flashcards = await process_with_gemini_latest(prompt, text)
        
        return ProcessingResponse(status="success", result=flashcards)
        
    except Exception as e:
        print(f"Flashcard generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import tempfile
import google.generativeai as genai
from supabase import create_client, Client
# import fitz  # PyMuPDF - Commented out for cloud deployment
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

class QuizResponse(BaseModel):
    status: str
    result: dict

class FlashcardResponse(BaseModel):
    status: str
    result: dict

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
    """Extract text from PDF using pdfplumber"""
    text = ""
    
    try:
        # Use pdfplumber for cloud deployment compatibility
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            return text
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")
    
    return text

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
    """Process text with latest Gemini 2.5 Pro model"""
    try:
        print(f"Starting Gemini 2.5 Pro processing...")
        print(f"Text length: {len(text)} characters")
        
        # Change the model name here to use Gemini 2.5 Pro
        model = genai.GenerativeModel('gemini-2.5-pro')

        # NOTE: You can now pass the full text, as 2.5 Pro has a large context window.
        # The chunking logic is no longer strictly necessary unless your files
        # are truly enormous, exceeding 1 million tokens.
        full_prompt = f"{prompt}\n\nDocument Content:\n{text}"
        response = model.generate_content(full_prompt)
        
        if response and response.text:
            result = response.text
            
            # Clean up JSON responses by removing markdown code blocks and extra formatting
            import re
            import json
            
            # First, try to extract JSON from markdown code blocks
            if "```json" in result or "```" in result:
                # Remove ```json and ``` markers
                result = re.sub(r'```json\s*', '', result)
                result = re.sub(r'```\s*$', '', result)
                result = result.strip()
                print(f"Cleaned response from markdown blocks")
            
            # Try to parse and return as JSON object for quiz/flashcard endpoints
            try:
                # Parse the JSON string into a Python object
                parsed_json = json.loads(result)
                print(f"Successfully parsed JSON response")
                return parsed_json  # Return as object, not string
            except json.JSONDecodeError:
                # If it's not valid JSON, return as cleaned text (for summaries, Q&A)
                result = result.strip()
                print(f"Non-JSON response, returning cleaned text")
                return result
            
            return result
        else:
            return "Empty response from AI"
        
    except Exception as e:
        print(f"Gemini 2.5 Pro error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # Firebase Admin SDK verification with enhanced error handling
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        error_str = str(e)
        print(f"Token verification error: {e}")
        
        # Handle specific timing errors with a simple retry
        if "Token used too early" in error_str or "Check that your computer's clock is set correctly" in error_str:
            try:
                # Simple retry for timing issues
                import time
                time.sleep(1)
                decoded_token = auth.verify_id_token(token)
                print("Token verification succeeded on retry")
                return decoded_token
            except Exception as retry_error:
                print(f"Token verification failed on retry: {retry_error}")
                raise HTTPException(status_code=401, detail=f"Invalid token: {retry_error}")
        else:
            # For other errors, fail immediately
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

# API Routes
@app.get("/")
async def root():
    return {"message": "PDF Study Assistant API", "version": "2.0.0", "model": "gemini-2.5-pro"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "supabase": bool(supabase),
            "gemini": bool(gemini_api_key),
            "firebase": bool(firebase_admin._apps)
        },
        "ai_model": "gemini-2.5-pro"
    }

@app.post("/upload-test")
async def upload_test(file: UploadFile = File(...)):
    """Simple upload test without authentication"""
    try:
        print(f"Upload test - Received file: {file.filename}")
        print(f"Upload test - File content type: {file.content_type}")
        print(f"Upload test - File size: {file.size}")
        
        # Just read the file to test
        content = await file.read()
        print(f"Upload test - Successfully read {len(content)} bytes")
        
        return {
            "status": "success",
            "message": "Test upload successful",
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type
        }
    except Exception as e:
        print(f"Upload test error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload test failed: {e}")

@app.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    user = Depends(verify_token)
):
    """Upload PDF to Supabase storage with improved duplicate handling"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        user_id = user['uid']
        file_path = f"user_uploads/{user_id}/{file.filename}"
        
        # First, check if user already has this file in Supabase storage
        try:
            # Try to download to check if exists
            existing_file_check = supabase.storage.from_("pdfs").download(file_path)
            if existing_file_check:
                print(f"File already exists for user {user_id}: {file.filename}")
                
                # Get public URL for existing file
                public_url_response = supabase.storage.from_("pdfs").get_public_url(file_path)
                
                # Handle both string and dict responses
                if isinstance(public_url_response, dict) and 'publicUrl' in public_url_response:
                    file_url = public_url_response['publicUrl']
                elif isinstance(public_url_response, str):
                    file_url = public_url_response
                else:
                    # Construct URL manually if response is unexpected
                    file_url = f"{supabase_url}/storage/v1/object/public/pdfs/{file_path}"
                
                # Add to memory for future use
                uploaded_files[file_path] = {
                    'filename': file.filename,
                    'file_url': file_url,
                    'size': len(existing_file_check) if existing_file_check else 0,
                    'uploaded_at': datetime.now().isoformat()
                }
                
                return PDFUploadResponse(
                    status="exists",
                    file_id=file_path,
                    filename=file.filename,
                    file_url=file_url
                )
        except Exception as check_error:
            # File doesn't exist, proceed with upload
            print(f"File doesn't exist for user {user_id}, proceeding with upload: {file.filename}")

        # Read file content
        file_content = await file.read()
        
        # Upload new file to Supabase
        try:
            upload_response = supabase.storage.from_("pdfs").upload(
                file_path, file_content,
                file_options={"content-type": "application/pdf"}
            )
            print(f"Successfully uploaded: {file_path}")
        except Exception as upload_error:
            # If upload fails due to duplicate (rare case), return existing file info
            if "already exists" in str(upload_error).lower() or "duplicate" in str(upload_error).lower():
                print(f"Upload failed due to duplicate, fetching existing file: {file_path}")
                public_url_response = supabase.storage.from_("pdfs").get_public_url(file_path)
                
                # Handle both string and dict responses
                if isinstance(public_url_response, dict) and 'publicUrl' in public_url_response:
                    file_url = public_url_response['publicUrl']
                elif isinstance(public_url_response, str):
                    file_url = public_url_response
                else:
                    # Construct URL manually if response is unexpected
                    file_url = f"{supabase_url}/storage/v1/object/public/pdfs/{file_path}"
                
                # Add to memory
                uploaded_files[file_path] = {
                    'filename': file.filename,
                    'file_url': file_url,
                    'size': len(file_content),
                    'uploaded_at': datetime.now().isoformat()
                }
                
                return PDFUploadResponse(
                    status="exists",
                    file_id=file_path,
                    filename=file.filename,
                    file_url=file_url
                )
            else:
                print(f"Upload failed: {upload_error}")
                raise upload_error
        
        # Get public URL for new upload
        public_url_response = supabase.storage.from_("pdfs").get_public_url(file_path)
        
        # Handle both string and dict responses
        if isinstance(public_url_response, dict) and 'publicUrl' in public_url_response:
            file_url = public_url_response['publicUrl']
        elif isinstance(public_url_response, str):
            file_url = public_url_response
        else:
            # Construct URL manually if response is unexpected
            file_url = f"{supabase_url}/storage/v1/object/public/pdfs/{file_path}"
        
        # Store in memory for quick access
        uploaded_files[file_path] = {
            'filename': file.filename,
            'file_url': file_url,
            'size': len(file_content),
            'uploaded_at': datetime.now().isoformat()
        }
        
        print(f"New file uploaded successfully: {file.filename} for user {user_id}")
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
    """Get list of user's uploaded documents with improved error handling"""
    try:
        user_id = user['uid']
        user_files = []
        user_folder = f"user_uploads/{user_id}"
        
        print(f"Fetching documents for user: {user_id}")
        
        # Get files from Supabase storage (primary source)
        try:
            storage_response = supabase.storage.from_("pdfs").list(user_folder)
            
            print(f"Supabase storage response type: {type(storage_response)}")
            print(f"Supabase storage response: {storage_response}")
            
            if isinstance(storage_response, list):
                for file_obj in storage_response:
                    if isinstance(file_obj, dict) and file_obj.get('name'):
                        file_path = f"user_uploads/{user_id}/{file_obj['name']}"
                        
                        # Get public URL
                        try:
                            public_url_response = supabase.storage.from_("pdfs").get_public_url(file_path)
                            
                            # Handle both string and dict responses
                            if isinstance(public_url_response, dict) and 'publicUrl' in public_url_response:
                                file_url = public_url_response['publicUrl']
                            elif isinstance(public_url_response, str):
                                file_url = public_url_response
                            else:
                                # Construct URL manually if response is unexpected
                                file_url = f"{supabase_url}/storage/v1/object/public/pdfs/{file_path}"
                            
                            # Safely get metadata
                            metadata = file_obj.get('metadata', {})
                            file_size = 0
                            if isinstance(metadata, dict):
                                file_size = metadata.get('size', 0)
                            
                            file_info = {
                                "id": file_path,
                                "filename": file_obj['name'],
                                "file_url": file_url,
                                "size": file_size,
                                "uploaded_at": file_obj.get('created_at', file_obj.get('updated_at', datetime.now().isoformat()))
                            }
                            
                            user_files.append(file_info)
                            
                            # Update memory cache
                            uploaded_files[file_path] = {
                                'filename': file_obj['name'],
                                'file_url': file_url,
                                'size': file_size,
                                'uploaded_at': file_obj.get('created_at', file_obj.get('updated_at', datetime.now().isoformat()))
                            }
                            
                        except Exception as url_error:
                            print(f"Error getting URL for file {file_obj['name']}: {url_error}")
                            continue
                            
                print(f"Found {len(user_files)} files for user {user_id}")
                            
        except Exception as storage_error:
            print(f"Error listing Supabase files: {storage_error}")
            print(f"Error type: {type(storage_error)}")
            
            # Fallback to in-memory storage if Supabase fails
            print("Falling back to in-memory storage...")
            for file_id, file_info in uploaded_files.items():
                if file_id.startswith(f"user_uploads/{user_id}/"):
                    user_files.append({
                        "id": file_id,
                        "filename": file_info['filename'],
                        "file_url": file_info['file_url'],
                        "size": file_info['size'],
                        "uploaded_at": file_info['uploaded_at']
                    })
        
        return {
            "status": "success",
            "data": {
                "pdfs": user_files,
                "count": len(user_files)
            }
        }
    except Exception as e:
        print(f"Error in get_documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {e}")

@app.get("/profile")
async def get_profile(user = Depends(verify_token)):
    """Get user profile with PDF count"""
    try:
        user_id = user['uid']
        user_email = user.get('email', 'Unknown')
        
        # Count user's PDFs
        pdf_count = 0
        
        # Count from in-memory storage
        for file_id in uploaded_files:
            if file_id.startswith(f"user_uploads/{user_id}/"):
                pdf_count += 1
        
        # Also count from Supabase storage
        try:
            user_folder = f"user_uploads/{user_id}"
            storage_response = supabase.storage.from_("pdfs").list(user_folder)
            
            if isinstance(storage_response, list):
                storage_count = len([f for f in storage_response if isinstance(f, dict) and f.get('name')])
                # Use max to avoid double counting
                pdf_count = max(pdf_count, storage_count)
        except Exception as storage_error:
            print(f"Error counting Supabase files: {storage_error}")
        
        return {
            "user_id": user_id,
            "email": user_email,
            "pdf_count": pdf_count,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {e}")

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

@app.post("/quiz", response_model=QuizResponse)
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
        Return ONLY valid JSON with this exact structure (no markdown, no code blocks):
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
        
        return QuizResponse(status="success", result=quiz)
        
    except Exception as e:
        print(f"Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {e}")

@app.post("/flashcards", response_model=FlashcardResponse)
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
        Return ONLY valid JSON with this exact structure (no markdown, no code blocks):
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
        
        return FlashcardResponse(status="success", result=flashcards)
        
    except Exception as e:
        print(f"Flashcard generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {e}")

@app.delete("/delete-pdf/{pdf_id:path}")
async def delete_pdf(pdf_id: str, user = Depends(verify_token)):
    """Delete a PDF document from storage"""
    try:
        user_id = user['uid']
        
        # Decode the file path (handle URL encoding)
        import urllib.parse
        file_path = urllib.parse.unquote(pdf_id)
        
        print(f"Delete request for file: {file_path}")
        print(f"User ID: {user_id}")
        
        # Verify the file belongs to the user (security check)
        expected_prefix = f"user_uploads/{user_id}/"
        if not file_path.startswith(expected_prefix):
            raise HTTPException(status_code=403, detail="Access denied: File does not belong to user")
        
        # Delete the file from Supabase storage
        try:
            storage_response = supabase.storage.from_("pdfs").remove([file_path])
            print(f"Storage deletion response: {storage_response}")
            
            # Also remove from memory cache if it exists
            if file_path in uploaded_files:
                del uploaded_files[file_path]
                
        except Exception as storage_error:
            print(f"Error deleting file from storage: {storage_error}")
            raise HTTPException(status_code=500, detail=f"Failed to delete file from storage: {storage_error}")
        
        return {"status": "success", "message": "PDF deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Delete PDF error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete PDF: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

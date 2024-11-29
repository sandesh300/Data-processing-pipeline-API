import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import requests
import json
from typing import Dict, Optional, Any
from sqlalchemy.orm import Session
from models import SessionLocal, ProcessedText
from local_llm import process_text_local

app = FastAPI()

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class TextInput(BaseModel):
    text: str
    context: Optional[str] = None
    style: Optional[str] = None

class TextOutput(BaseModel):
    field1: str
    field2: str
    prompt_used: str
    model_name: str

    class Config:
        orm_mode = True

class ComparisonOutput(BaseModel):
    field1_match: bool
    field2_match: bool
    api_response: TextOutput
    local_response: TextOutput
    similarity_score: float
    details: Dict[str, Any]
    context_analysis: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True

def build_gemini_prompt(text: str, context: Optional[str] = None, style: Optional[str] = None) -> str:
    """Build a structured prompt for Gemini API"""
    base_prompt = "You are a helpful AI assistant. "
    
    if style:
        base_prompt += f"Please respond in a {style} style. "
    
    if context:
        base_prompt += f"Context: {context}\n"
    
    base_prompt += f"Question: {text}\n"
    base_prompt += "Please provide a clear, structured response."
    
    return base_prompt

def build_local_prompt(text: str, context: Optional[str] = None, style: Optional[str] = None) -> str:
    """Build a structured prompt for local LLM"""
    base_prompt = "Act as an AI expert. "
    
    if style:
        base_prompt += f"Respond in {style} style. "
    
    if context:
        base_prompt += f"Given this context: {context}\n"
    
    base_prompt += f"Question: {text}\n"
    base_prompt += "Provide a detailed analysis."
    
    return base_prompt

def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

@app.post("/process-text/", response_model=TextOutput)
async def process_text(input_data: TextInput, db: Session = Depends(get_db)):
    try:
        prompt = build_gemini_prompt(input_data.text, input_data.context, input_data.style)
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        api_response = response.json()
        generated_text = api_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        output_data = TextOutput(
            field1=generated_text[:100],
            field2=generated_text[100:200] if len(generated_text) > 100 else "",
            prompt_used=prompt,
            model_name="gemini-pro"
        )
        
        db_text = ProcessedText(
            original_text=input_data.text,
            field1=output_data.field1,
            field2=output_data.field2,
            prompt_used=prompt
        )
        db.add(db_text)
        db.commit()
        
        return output_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"API processing failed: {str(e)}"
        )

@app.post("/process-text-local/", response_model=TextOutput)
async def process_text_local_endpoint(input_data: TextInput, db: Session = Depends(get_db)):
    try:
        prompt = build_local_prompt(input_data.text, input_data.context, input_data.style)
        local_response = process_text_local(prompt)
        
        if not isinstance(local_response, str):
            generated_text = str(local_response)
        else:
            generated_text = local_response
        
        output_data = TextOutput(
            field1=generated_text[:100] if generated_text else "",
            field2=generated_text[100:200] if len(generated_text) > 100 else "",
            prompt_used=prompt,
            model_name="local-llm"
        )
        
        db_text = ProcessedText(
            original_text=input_data.text,
            field1=output_data.field1,
            field2=output_data.field2,
            prompt_used=prompt
        )
        db.add(db_text)
        db.commit()
        
        return output_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Local processing failed: {str(e)}"
        )

@app.post("/compare-outputs/", response_model=ComparisonOutput)
async def compare_outputs_endpoint(input_data: TextInput, db: Session = Depends(get_db)):
    try:
        api_response = await process_text(input_data, db)
        local_response = await process_text_local_endpoint(input_data, db)
        
        comparison_report = ComparisonOutput(
            field1_match=api_response.field1 == local_response.field1,
            field2_match=api_response.field2 == local_response.field2,
            api_response=api_response,
            local_response=local_response,
            similarity_score=calculate_similarity_score(
                api_response.field1 + api_response.field2,
                local_response.field1 + local_response.field2
            ),
            details={
                "prompts_used": {
                    "api": api_response.prompt_used,
                    "local": local_response.prompt_used
                },
                "response_lengths": {
                    "api": len(api_response.field1) + len(api_response.field2),
                    "local": len(local_response.field1) + len(local_response.field2)
                }
            }
        )
        
        return comparison_report
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Comparison failed",
                "input_text": input_data.text
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


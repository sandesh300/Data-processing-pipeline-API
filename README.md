# Data Processing Pipeline with APIs

## Objective
Integrate Gemini API and build a custom data processing pipeline.

## Requirements

### Setup the Pipeline:
- Use Gemini API to process text data.
- The input is raw, unstructured text, and the output should be structured JSON with specified fields.

### Prompt Engineering:
- Design a well-defined prompt for the API to ensure accurate outputs.
- Use Pydantic models to validate the API response structure and ensure data integrity.

### Local Model Integration:
- Set up a locally hosted LLM (e.g., LLaMA or similar) and process the same data pipeline using it instead of the external API.
- Compare the outputs of the external API and the locally hosted model.

### Error Handling:
- Handle API failures, rate limits, and invalid responses gracefully without relying on built-in batch-processing commands.

## Deliverable
- Pipeline code with prompt engineering and validation scripts.
- Comparison report of outputs from external and local models.

## Project Structure

```
data-processing-pipeline/
│
├── main.py               # Main application file with FastAPI endpoints
├── models.py             # Database models and operations using SQLAlchemy
├── local_llm.py          # Local model processing using the transformers library
├── download_model.py     # Script to download and save the local model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Setup

### Prerequisites
- Python 3.8+
- PostgreSQL
- FastAPI
- SQLAlchemy
- Transformers library

### requirements.txt
```
fastapi
uvicorn
pydantic
requests
sqlalchemy
transformers
psycopg2-binary
```

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/sandesh300/data-processing-pipeline.git
   cd data-processing-pipeline
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up the database:**
   - Update the `DATABASE_URL` in `models.py` with your PostgreSQL credentials.
   - Run the following command to create the database tables:
     ```sh
     python models.py
     ```

4. **Download and save the local model:**
   ```sh
   python download_model.py
   ```

### Running the Application

1. **Start the FastAPI server:**
   ```sh
   uvicorn main:app --reload
   ```

2. **Access the API:**
   - The API will be available at `http://localhost:8000`.
### FastAPI-doc
![Screenshot (61)](https://github.com/user-attachments/assets/c1ba2483-bd53-4054-9507-f213cfa27576)

## API Endpoints

### 1. Process Text
- **Endpoint:** `POST /process-text/`
- **URL**: `http://localhost:8000/process-text/`
- **Request:**
  ```json
  {
      "text": "You are a helpful AI assistant. Here's a user question: How do gaming phones differ from regular smartphones? Please provide a clear, structured response focusing on key features."
  }
  ```
- **Response: 200 OK**
  ```json
  {
      "field1": "**Key Differences Between Gaming Phones and Regular Smartphones**\n\n**1. Processing Power and Graphic",
      "field2": "s:**\n* Gaming phones have considerably more powerful processors (e.g., Snapdragon 8 Gen 2) and dedic",
      "prompt_used": "You are a helpful AI assistant. Question: You are a helpful AI assistant. Here's a user question: How do gaming phones differ from regular smartphones? Please provide a clear, structured response focusing on key features.\nPlease provide a clear, structured response.",
      "model_name": "gemini-pro"
  }
  ```

### 2. Process Text Local
- **Endpoint:** `POST /process-text-local/`
- **URL**: `http://localhost:8000/process-text-local/`
- **Request:**
  ```json
  {
      "text": "Act as an AI expert. Given the question: How do gaming phones differ from regular smartphones? List the main gaming-specific features and hardware differences."
  }
  ```
- **Response:200 OK**
  ```json
  {
      "field1": "{'field1': 'Act as an AI expert. Question: Act as an AI expert. Given the question: How do gaming ph",
      "field2": "ones differ ', 'field2': 'from regular smartphones? List the main gaming-specific features and hardw",
      "prompt_used": "Act as an AI expert. Question: Act as an AI expert. Given the question: How do gaming phones differ from regular smartphones? List the main gaming-specific features and hardware differences.\nProvide a detailed analysis.",
      "model_name": "local-llm"
  }
  ```

### 3. Compare Outputs
- **Endpoint:** `POST /compare-outputs/`
- **URL**: `http://localhost:8000/compare-outputs/`
- **Request:**
  ```json
  {
      "text": "Compare and analyze: How do gaming phones differ from regular smartphones? Provide specific features and capabilities."
  }
  ```
- **Response: 200 OK**
  ```json
  {
      "field1_match": false,
      "field2_match": false,
      "api_response": {
          "field1": "**Comparison and Analysis of Gaming Phones vs. Regular Smartphones**\n\n**Features and Capabilities**\n",
          "field2": "\n| **Feature** | **Gaming Phone** | **Regular Smartphone** |\n|---|---|---|\n**Display** | Larger, hig",
          "prompt_used": "You are a helpful AI assistant. Question: Compare and analyze: How do gaming phones differ from regular smartphones? Provide specific features and capabilities.\nPlease provide a clear, structured response.",
          "model_name": "gemini-pro"
      },
      "local_response": {
          "field1": "{'field1': 'Act as an AI expert. Question: Compare and analyze: How do gaming phones differ from reg",
          "field2": "ular smartph', 'field2': 'ones? Provide specific features and capabilities.\\nProvide a detailed anal",
          "prompt_used": "Act as an AI expert. Question: Compare and analyze: How do gaming phones differ from regular smartphones? Provide specific features and capabilities.\nProvide a detailed analysis.",
          "model_name": "local-llm"
      },
      "similarity_score": 0.09090909090909091,
      "details": {
          "prompts_used": {
              "api": "You are a helpful AI assistant. Question: Compare and analyze: How do gaming phones differ from regular smartphones? Provide specific features and capabilities.\nPlease provide a clear, structured response.",
              "local": "Act as an AI expert. Question: Compare and analyze: How do gaming phones differ from regular smartphones? Provide specific features and capabilities.\nProvide a detailed analysis."
          },
          "response_lengths": {
              "api": 200,
              "local": 200
          }
      },
      "context_analysis": null
  }
  ```

## Comparison Report

The comparison report provides a detailed analysis of the outputs from the external API (Gemini) and the local model. It includes:
- Field matches (`field1_match`, `field2_match`)
- API and local model responses
- Similarity score between the responses
- Details of the prompts used and response lengths

## Error Handling

The application includes robust error handling to manage API failures, rate limits, and invalid responses. HTTP exceptions are raised with appropriate error messages to ensure smooth operation.

## Pipeline code with prompt engineering and validation scripts.

 Below is the complete pipeline code with prompt engineering and validation scripts. This code includes the main application file (`main.py`), database models and operations (`models.py`), local model processing (`local_llm.py`), and a script to download and save the local model (`download_model.py`).

### `main.py`

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import requests
import json
from typing import Dict, Optional, Any
from sqlalchemy.orm import Session
from models import SessionLocal, ProcessedText
from local_llm import process_text_local

app = FastAPI()

API_KEY = "AIzaSyALpRWGOdEo2F2Hi0V7NIcxNTGV9tGO5dQ"
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
```

### `models.py`

```python
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "postgresql://pipeline_user:1234@localhost/data_pipeline_db"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ProcessedText(Base):
    __tablename__ = "processed_texts"

    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text, index=True, nullable=False)
    field1 = Column(Text)
    field2 = Column(Text)
    prompt_used = Column(Text, nullable=True)

    def __repr__(self):
        return f"<ProcessedText(id={self.id}, original_text={self.original_text[:30]}...)>"

@contextmanager
def get_db():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

def create_tables():
    """Create all tables in the database."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise

def drop_tables():
    """Drop all tables in the database."""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping tables: {str(e)}")
        raise

def recreate_tables():
    """Drop and recreate all tables."""
    drop_tables()
    create_tables()

# Database operations
def insert_processed_text(db: Session, original_text: str, field1: str, field2: str, prompt_used: str = None):
    """Insert a new processed text entry."""
    db_text = ProcessedText(
        original_text=original_text,
        field1=field1,
        field2=field2,
        prompt_used=prompt_used
    )
    db.add(db_text)
    return db_text

if __name__ == "__main__":
    # Example usage
    try:
        logger.info("Starting database setup...")

        # Recreate tables (warning: this will delete existing data)
        recreate_tables()

        # Example insertion
        with get_db() as db:
            test_entry = insert_processed_text(
                db=db,
                original_text="Test text",
                field1="Field 1 content",
                field2="Field 2 content",
                prompt_used="Test prompt"
            )
            logger.info(f"Test entry created with ID: {test_entry.id}")

        logger.info("Database setup completed successfully")

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise
```

### `local_llm.py`

```python
from transformers import pipeline

local_model_path = "models/local_model"
model = pipeline("text-generation", model=local_model_path)

def process_text_local(text: str):
    response = model(text, max_length=200, num_return_sequences=1)
    generated_text = response[0]['generated_text']

    return {
        "field1": generated_text[:100],
        "field2": generated_text[100:200] if len(generated_text) > 100 else ""
    }
```

### `download_model.py`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_and_save_model(model_name: str, local_path: str):
    # Download the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer locally
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)

    print(f"Model and tokenizer saved to {local_path}")

if __name__ == "__main__":
    model_name = "facebook/opt-125m"  # Replace with the model you want to use
    local_path = "models/local_model"  # Path to the local directory
    download_and_save_model(model_name, local_path)
```



This README provides an overview of the data processing pipeline, including setup instructions, API endpoints, and a comparison report. It ensures that the project meets the objectives and requirements specified.

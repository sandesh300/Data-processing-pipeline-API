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
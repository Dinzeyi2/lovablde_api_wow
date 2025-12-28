from fastapi import HTTPException, Header
from typing import Optional
import secrets
import hashlib
from datetime import datetime
from app.models import APIKey

def generate_api_key() -> str:
    """Generate a secure API key"""
    random_bytes = secrets.token_bytes(32)
    api_key = hashlib.sha256(random_bytes).hexdigest()
    return f"algoapi_{api_key[:32]}"

def create_api_key(db, name: str, email: str) -> APIKey:
    """Create a new API key in the database"""
    api_key = generate_api_key()
    user_id = hashlib.sha256(email.encode()).hexdigest()[:16]
    
    db_api_key = APIKey(
        key=api_key,
        user_id=user_id,
        name=name,
        email=email
    )
    
    db.add(db_api_key)
    db.commit()
    db.refresh(db_api_key)
    
    return db_api_key

def verify_api_key(x_api_key: Optional[str] = Header(None), db = None) -> APIKey:
    """Verify API key from request header"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing")
    
    # Import here to avoid circular dependency
    from app.main import get_db
    if db is None:
        db = next(get_db())
    
    api_key = db.query(APIKey).filter(APIKey.key == x_api_key, APIKey.is_active == True).first()
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Update last used timestamp
    api_key.last_used = datetime.utcnow()
    db.commit()
    
    return api_key

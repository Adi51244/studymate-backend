from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
import hashlib
import time
import firebase_admin
from firebase_admin import auth

# Create a security instance
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # If no credentials provided, return anonymous user
    if credentials is None:
        print('No credentials provided: Using anonymous user')
        return {'uid': 'anonymous_user', 'email': 'anonymous@studymate.com'}
    
    token = credentials.credentials
    
    # Check if Firebase Admin is available
    firebase_available = False
    try:
        firebase_available = bool(firebase_admin._apps)
    except:
        firebase_available = False
    
    # Try Firebase Admin SDK first if available
    if firebase_available:
        try:
            # Firebase Admin SDK verification
            decoded_token = auth.verify_id_token(token)
            print("Firebase Admin SDK: Verified token successfully")
            return decoded_token
        except Exception as e:
            error_str = str(e)
            print(f'Firebase token verification error: {e}')
            
            # Handle specific timing errors with a simple retry
            if 'Token used too early' in error_str or 'Check that your computer' in error_str:
                try:
                    # Simple retry for timing issues
                    time.sleep(1)
                    decoded_token = auth.verify_id_token(token)
                    print('Token verification succeeded on retry')
                    return decoded_token
                except Exception as retry_e:
                    print(f'Token verification retry failed: {retry_e}')
    
    # Fallback to auth_helper for production mode
    try:
        from auth_helper import get_user_from_token
        print("Using auth_helper for token verification")
        return get_user_from_token(token, firebase_available=firebase_available)
    except ImportError:
        print('auth_helper not available, using final fallback')
        
    # Last resort fallback
    user_hash = hashlib.md5(token.encode()).hexdigest()[:8]
    fallback_user_id = f"user_{user_hash}"
    print(f"Final fallback: Using generated user {fallback_user_id}")
    return {"uid": fallback_user_id, "email": f"{fallback_user_id}@studymate.com"}

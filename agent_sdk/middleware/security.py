import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, ExpiredSignatureError, jwt as _jwt
from fastapi import Request, HTTPException, status

logger = logging.getLogger("agent_sdk.security")

class JWTAuth:
    def __init__(self, secret: str = None, algorithm: str = "HS256"):
        self.secret = secret or os.getenv("AUTH_JWT_SECRET")
        self.algorithm = algorithm
        if not self.secret:
            # In dev, we might allow it, but in prod it's critical
            logger.warning("AUTH_JWT_SECRET not set. JWT validation will fail.")

    def decode_access_token(self, token: str) -> Optional[str]:
        """Decode an access token and return the subject (user_id)."""
        if not self.secret:
            return None
        try:
            payload = _jwt.decode(token, self.secret, algorithms=[self.algorithm])
            if payload.get("type") != "access":
                return None
            return payload.get("sub")
        except ExpiredSignatureError:
            return None
        except JWTError as e:
            logger.warning("JWT decode error: %s", type(e).__name__)
            return None

    def decode_refresh_token(self, token: str) -> Optional[str]:
        """Decode a refresh token and return the subject (user_id)."""
        if not self.secret:
            return None
        try:
            payload = _jwt.decode(token, self.secret, algorithms=[self.algorithm])
            if payload.get("type") != "refresh":
                return None
            return payload.get("sub")
        except JWTError:
            return None

    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=1))
        return _jwt.encode({"sub": user_id, "exp": expire, "type": "access"}, self.secret, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=7))
        return _jwt.encode({"sub": user_id, "exp": expire, "type": "refresh"}, self.secret, algorithm=self.algorithm)

def get_user_from_header(request: Request, auth_manager: JWTAuth) -> Optional[str]:
    """Helper to extract and decode user_id from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        return None
    return auth_manager.decode_access_token(token)

"""Authentication module - JWT + API key management."""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
from jose import JWTError, jwt


@dataclass
class TokenData:
    user_id: str
    email: str


class AuthManager:
    """JWT + API Key authentication manager."""

    def __init__(self, secret_key: str = "change-me-in-production",
                 algorithm: str = "HS256", expiry_hours: int = 24):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiry_hours = expiry_hours

    def create_token(self, user_id: str, email: str) -> str:
        """Create a JWT access token."""
        expire = datetime.utcnow() + timedelta(hours=self.expiry_hours)
        payload = {
            "sub": user_id,
            "email": email,
            "exp": expire,
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            email = payload.get("email")
            if user_id is None:
                return None
            return TokenData(user_id=user_id, email=email)
        except JWTError:
            return None

    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}:{hashed}"

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, stored_hash = hashed.split(":")
            computed = hashlib.sha256((salt + password).encode()).hexdigest()
            return computed == stored_hash
        except ValueError:
            return False

    def generate_api_key(self) -> tuple[str, str]:
        """Generate a new API key pair (key, hash).

        Returns:
            Tuple of (raw_key, hashed_key) - raw_key shown once to user
        """
        raw_key = f"acms_{secrets.token_hex(24)}"
        hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()
        return raw_key, hashed_key

    def verify_api_key(self, raw_key: str, stored_hash: str) -> bool:
        """Verify an API key against its stored hash."""
        computed = hashlib.sha256(raw_key.encode()).hexdigest()
        return computed == stored_hash

    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate a user (placeholder - in production, query DB).

        Returns:
            User dict if authenticated, None otherwise
        """
        # Placeholder: in production, query database
        # For now, accept any credentials for development
        return {
            "id": "user_dev_001",
            "email": email,
        }

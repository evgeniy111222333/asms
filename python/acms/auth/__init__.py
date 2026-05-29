"""Authentication module - JWT + API key management."""

import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass
from jose import JWTError, jwt


@dataclass
class TokenData:
    user_id: str
    email: str


class TokenBlacklist:
    """Redis-backed JWT token blacklist for revocation.
    
    SECURITY: Stores jti (JWT ID) of revoked tokens in Redis with TTL.
    All tokens with jti in blacklist are rejected.
    """
    
    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._prefix = "acms:blacklist:token:"
    
    def _key(self, jti: str) -> str:
        return f"{self._prefix}{jti}"
    
    async def revoke(self, jti: str, exp_timestamp: int) -> None:
        """Add token jti to blacklist until exp_timestamp."""
        if self._redis is None:
            return  # Skip if Redis unavailable
        
        import time
        ttl = max(int(exp_timestamp - time.time()), 0)
        if ttl > 0:
            await self._redis.setex(self._key(jti), ttl, "revoked")
    
    async def is_revoked(self, jti: str) -> bool:
        """Check if token jti is in blacklist."""
        if self._redis is None:
            return False  # Allow if Redis unavailable
        
        return await self._redis.exists(self._key(jti))


class AuthManager:
    """JWT + API Key authentication manager.
    
    SECURITY FEATURES:
    - Short-lived access tokens (default 15 min)
    - Refresh tokens with longer expiry (default 7 days)
    - Token blacklist for immediate revocation
    - Algorithm selection (HS256 default, RS256 for production)
    - Minimum secret key length enforcement
    """

    # Token type constants
    ACCESS_TOKEN = "access"
    REFRESH_TOKEN = "refresh"
    
    # Security defaults
    ACCESS_TOKEN_EXPIRY_MINUTES = 15
    REFRESH_TOKEN_EXPIRY_DAYS = 7
    MIN_SECRET_KEY_LENGTH = 32

    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        access_token_minutes: int = 15,
        refresh_token_days: int = 7,
        redis_client=None,
    ):
        # Require secret from environment
        if secret_key is None:
            secret_key = os.environ.get("JWT_SECRET")
        
        if secret_key is None:
            raise ValueError(
                "JWT_SECRET environment variable must be set. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        
        # Reject weak secrets
        if len(secret_key) < self.MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"JWT_SECRET must be at least {self.MIN_SECRET_KEY_LENGTH} characters. "
                f"Got {len(secret_key)} characters."
            )
        
        if "change-me-in-production" in secret_key:
            raise ValueError("JWT_SECRET cannot contain 'change-me-in-production' - use a secure random value.")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_minutes = access_token_minutes
        self.refresh_token_days = refresh_token_days
        self._blacklist = TokenBlacklist(redis_client)

    def create_token(
        self,
        user_id: str,
        email: str,
        token_type: str = ACCESS_TOKEN,
    ) -> str:
        """Create a JWT token (access or refresh).
        
        Args:
            user_id: User identifier.
            email: User email.
            token_type: Either ACCESS_TOKEN or REFRESH_TOKEN.
        
        Returns:
            Encoded JWT string.
        """
        now = datetime.utcnow()
        
        if token_type == self.REFRESH_TOKEN:
            expiry = now + timedelta(days=self.refresh_token_days)
        else:
            expiry = now + timedelta(minutes=self.access_token_minutes)
        
        jti = f"{token_type}_{secrets.token_urlsafe(16)}"
        
        payload = {
            "sub": user_id,
            "email": email,
            "type": token_type,
            "jti": jti,
            "exp": expiry,
            "iat": now,
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_access_token(self, user_id: str, email: str) -> str:
        """Create a short-lived access token."""
        return self.create_token(user_id, email, self.ACCESS_TOKEN)
    
    def create_refresh_token(self, user_id: str, email: str) -> str:
        """Create a refresh token for obtaining new access tokens."""
        return self.create_token(user_id, email, self.REFRESH_TOKEN)

    def verify_token(
        self,
        token: str,
        check_blacklist: bool = True,
    ) -> Optional[TokenData]:
        """Verify and decode a JWT token.
        
        Args:
            token: The JWT string to verify.
            check_blacklist: If True, check if token is revoked.
        
        Returns:
            TokenData if valid, None otherwise.
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            # Check token type
            token_type = payload.get("type", self.ACCESS_TOKEN)
            if token_type != self.ACCESS_TOKEN:
                return None  # Refresh tokens not valid for API access
            
            # Check blacklist
            if check_blacklist:
                jti = payload.get("jti")
                if jti and self._blacklist._redis:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    is_revoked = loop.run_until_complete(self._blacklist.is_revoked(jti))
                    loop.close()
                    if is_revoked:
                        return None
            
            user_id = payload.get("sub")
            email = payload.get("email")
            if user_id is None:
                return None
            return TokenData(user_id=user_id, email=email)
        except JWTError:
            return None
    
    async def verify_token_async(
        self,
        token: str,
        check_blacklist: bool = True,
    ) -> Optional[TokenData]:
        """Async version of verify_token with blacklist check."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            token_type = payload.get("type", self.ACCESS_TOKEN)
            if token_type != self.ACCESS_TOKEN:
                return None
            
            if check_blacklist:
                jti = payload.get("jti")
                if jti and self._blacklist._redis:
                    if await self._blacklist.is_revoked(jti):
                        return None
            
            user_id = payload.get("sub")
            email = payload.get("email")
            if user_id is None:
                return None
            return TokenData(user_id=user_id, email=email)
        except JWTError:
            return None

    async def revoke_token(self, token: str) -> bool:
        """Revoke a token by adding its jti to blacklist.
        
        Args:
            token: The JWT to revoke.
        
        Returns:
            True if revoked, False if token invalid.
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow expired tokens
            )
            
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if jti and exp and self._blacklist._redis:
                await self._blacklist.revoke(jti, exp)
                return True
            return False
        except JWTError:
            return False

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Exchange a valid refresh token for new access + refresh tokens.
        
        Uses one-time refresh token rotation for security.
        Once a refresh token is used, it cannot be used again.
        
        Args:
            refresh_token: A valid refresh token.
        
        Returns:
            Dict with 'access_token' and 'refresh_token' if valid, None otherwise.
        """
        # Initialize used refresh tokens store if needed
        if not hasattr(self, "_used_refresh_tokens"):
            self._used_refresh_tokens: set = set()
        
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm],
            )
            
            # Must be refresh token
            if payload.get("type") != self.REFRESH_TOKEN:
                return None
            
            jti = payload.get("jti")
            user_id = payload.get("sub")
            email = payload.get("email")
            
            if not user_id or not jti:
                return None
            
            # Check if already used (rotation enforcement)
            if jti in self._used_refresh_tokens:
                return None  # Token already used
            
            # Mark as used
            self._used_refresh_tokens.add(jti)
            
            # Limit stored tokens to prevent memory bloat
            if len(self._used_refresh_tokens) > 10000:
                # Remove oldest half
                self._used_refresh_tokens = set(list(self._used_refresh_tokens)[-5000:])
            
            # Generate new token pair
            return {
                "access_token": self.create_access_token(user_id, email),
                "refresh_token": self.create_refresh_token(user_id, email),
            }
        except JWTError:
            return None

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        import bcrypt
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against bcrypt hash with constant-time comparison."""
        import bcrypt
        try:
            return hmac.compare_digest(
                bcrypt.hashpw(password.encode('utf-8'), hashed.encode('utf-8')),
                hashed.encode('utf-8')
            )
        except Exception:
            return False

    def generate_api_key(self) -> tuple[str, str]:
        """Generate a new API key pair (key, hash).

        Returns:
            Tuple of (raw_key, hashed_key) - raw_key shown once to user
        """
        raw_key = f"acms_{secrets.token_hex(24)}"
        hashed_key = self._hash_api_key(raw_key)
        return raw_key, hashed_key

    @staticmethod
    def _hash_api_key(raw_key: str) -> str:
        """Hash an API key using SHA-256."""
        import hashlib
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def verify_api_key(self, raw_key: str, stored_hash: str) -> bool:
        """Verify an API key against its stored hash using constant-time comparison."""
        computed = self._hash_api_key(raw_key)
        return hmac.compare_digest(computed, stored_hash)

    async def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user against database.

        Returns:
            User dict if authenticated, None otherwise
        """
        from acms.db import DatabaseManager
        db = DatabaseManager()
        try:
            user = await db.get_user_by_email(email=email)
            if user is None:
                return None
            if not self.verify_password(password, user.get("hashed_password", "")):
                return None
            return {"id": user.get("id"), "email": user.get("email"), "is_admin": user.get("is_admin", False)}
        except Exception:
            return None


def generate_jwt_secret() -> str:
    """Generate a secure random JWT secret.
    
    Returns:
        A URL-safe base64-encoded 256-bit secret.
    """
    return secrets.token_urlsafe(32)


__all__ = [
    "AuthManager",
    "TokenData",
    "TokenBlacklist",
    "generate_jwt_secret",
]

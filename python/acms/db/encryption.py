"""Credential encryption for exchange API keys."""

import base64
import hashlib
import os
import secrets


class CredentialEncryptor:
    """Encrypt/decrypt exchange credentials using Fernet symmetric encryption.
    
    SECURITY: Requires a strong key (minimum 32 characters). If a weak or
    default key is provided, encryption will fail to prevent accidental
    deployment with insecure defaults.
    """

    MIN_KEY_LENGTH = 32
    DEFAULT_KEY_PREFIX = "default-encryption-key"

    def __init__(self, key: str = None):
        from cryptography.fernet import Fernet
        
        if key is None:
            key = os.environ.get("ENCRYPTION_KEY")
        
        if key is None:
            raise ValueError(
                "ENCRYPTION_KEY environment variable must be set. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        
        # Check for default/weak keys
        if key.startswith(self.DEFAULT_KEY_PREFIX) or len(key) < self.MIN_KEY_LENGTH:
            raise ValueError(
                f"ENCRYPTION_KEY must be at least {self.MIN_KEY_LENGTH} characters "
                f"and cannot be a default key. Got key starting with: {key[:20]}..."
            )
        
        key_bytes = hashlib.sha256(key.encode()).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(key_bytes))

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string.

        Args:
            plaintext: The string to encrypt.

        Returns:
            Base64-encoded ciphertext string.
        """
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string.

        Args:
            ciphertext: The base64-encoded ciphertext to decrypt.

        Returns:
            Decrypted plaintext string.
        """
        return self._fernet.decrypt(ciphertext.encode()).decode()


def generate_encryption_key() -> str:
    """Generate a secure random encryption key.
    
    Returns:
        A URL-safe base64-encoded 256-bit key.
    """
    return secrets.token_urlsafe(32)


__all__ = [
    "CredentialEncryptor",
    "generate_encryption_key",
]

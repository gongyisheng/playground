from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair

def test_jwt():
    # Generate a key pair for testing
    key_pair = RSAKeyPair.generate()

    # Configure your server with the public key
    verifier = JWTVerifier(
        public_key=key_pair.public_key,
        issuer="https://test.yourcompany.com",
        audience="test-mcp-server"
    )

    # Generate a test token using the private key
    test_token = key_pair.create_token(
        subject="test-user-123",
        issuer="https://test.yourcompany.com", 
        audience="test-mcp-server",
        scopes=["read", "write", "admin"]
    )

    print(f"Test token: {test_token}")

if __name__ == "__main__":
    test_jwt()
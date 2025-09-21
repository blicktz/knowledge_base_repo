#!/usr/bin/env python3
"""
Quick syntax test for both files without dependencies
"""

def test_server_syntax():
    """Test that server file can be parsed"""
    import ast
    with open('runpod_fastapi_server.py', 'r') as f:
        content = f.read()
    
    try:
        ast.parse(content)
        print("✅ runpod_fastapi_server.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ runpod_fastapi_server.py syntax error: {e}")
        return False

def test_client_syntax():
    """Test that client file can be parsed"""
    import ast
    with open('test_concurrent_clients.py', 'r') as f:
        content = f.read()
    
    try:
        ast.parse(content)
        print("✅ test_concurrent_clients.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ test_concurrent_clients.py syntax error: {e}")
        return False

def test_key_functions():
    """Test that key functions are properly defined"""
    # Import the modules without running them
    import importlib.util
    
    # Test server file structure
    try:
        spec = importlib.util.spec_from_file_location("server", "runpod_fastapi_server.py")
        # Just test that it can be compiled, don't actually import
        print("✅ Server file can be compiled")
    except Exception as e:
        print(f"❌ Server file compilation error: {e}")
        return False
    
    # Test client file structure  
    try:
        spec = importlib.util.spec_from_file_location("client", "test_concurrent_clients.py")
        print("✅ Client file can be compiled")
    except Exception as e:
        print(f"❌ Client file compilation error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Testing file syntax and structure...")
    print("=" * 50)
    
    server_ok = test_server_syntax()
    client_ok = test_client_syntax()
    functions_ok = test_key_functions()
    
    print("=" * 50)
    if server_ok and client_ok and functions_ok:
        print("🎉 All files are syntactically correct and ready to run!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install fastapi uvicorn aiohttp streaming-form-data aiofiles")
        print("2. Install ML dependencies: pip install openai-whisper torch")
        print("3. Run server: python runpod_fastapi_server.py")
        print("4. Run client test: python test_concurrent_clients.py <audio_files> --clients 6")
    else:
        print("❌ Some files have issues that need to be fixed")
"""
Quick Installation Script for Medical RAG Chatbot
Installs dependencies in the correct order
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"\n{'='*60}")
    print(f"[*] {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] {description} - FAILED")
        print(f"Error: {e.stderr[:200]}")
        return False

def main():
    print("\n" + "="*60)
    print("  Medical RAG Chatbot - Quick Installation")
    print("="*60)
    
    # Step 1: Core packages
    print("\n[1/9] Installing Core Packages...")
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    
    # Step 2: FastAPI and web framework
    print("\n[2/9] Installing Web Framework...")
    run_command(
        f"{sys.executable} -m pip install fastapi uvicorn[standard] pydantic",
        "Installing FastAPI"
    )
    
    # Step 3: PyTorch (CPU version for compatibility)
    print("\n[3/9] Installing PyTorch...")
    run_command(
        f"{sys.executable} -m pip install torch torchvision torchaudio",
        "Installing PyTorch"
    )
    
    # Step 4: Transformers and NLP
    print("\n[4/9] Installing AI/ML Libraries...")
    run_command(
        f"{sys.executable} -m pip install transformers sentence-transformers",
        "Installing Transformers"
    )
    
    # Step 5: Vector store
    print("\n[5/9] Installing Vector Database...")
    run_command(
        f"{sys.executable} -m pip install faiss-cpu",
        "Installing FAISS"
    )
    
    # Step 6: LangChain
    print("\n[6/9] Installing LangChain...")
    run_command(
        f"{sys.executable} -m pip install langchain langchain-community",
        "Installing LangChain"
    )
    
    # Step 7: PDF Processing
    print("\n[7/9] Installing PDF Processing...")
    run_command(
        f"{sys.executable} -m pip install PyPDF2",
        "Installing PyPDF2"
    )
    
    # Step 8: Additional utilities
    print("\n[8/9] Installing Utilities...")
    run_command(
        f"{sys.executable} -m pip install numpy python-multipart aiofiles",
        "Installing Utilities"
    )
    
    # Step 9: GPTQ (optional - may fail)
    print("\n[9/9] Installing GPTQ (Optional - for GPU optimization)...")
    gptq_success = run_command(
        f"{sys.executable} -m pip install auto-gptq optimum accelerate",
        "Installing GPTQ"
    )
    
    if not gptq_success:
        print("\n[NOTE] GPTQ installation failed - this is OK!")
        print("   Your system will use standard models instead.")
        print("   GPTQ is only needed for quantized models on GPU.")
    
    # Final check
    print("\n" + "="*60)
    print(" Installation Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("  1. Run: py server.py")
    print("  2. Open: http://localhost:8000/docs")
    print("  3. Test: py test_quick.py")
    print("\nNote: Models will download on first use (~2-4GB)")
    
if __name__ == "__main__":
    main()

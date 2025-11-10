"""
Medical RAG Chatbot - Installation & Setup Script
Run this script first to set up your environment
"""
import os
import sys
import subprocess


def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'data/faiss_index',
        'data/uploads',
        'models',
        'utils',
        'app'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")


def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        sys.exit(1)


def verify_dataset():
    """Verify that the dataset directory exists"""
    dataset_path = "HackACure-Dataset/Dataset"
    if os.path.exists(dataset_path):
        pdf_files = [f for f in os.listdir(dataset_path) if f.endswith('.pdf')]
        print(f"\n✓ Found dataset with {len(pdf_files)} PDF files")
        for pdf in pdf_files[:5]:  # Show first 5
            print(f"  - {pdf}")
        if len(pdf_files) > 5:
            print(f"  ... and {len(pdf_files) - 5} more")
    else:
        print(f"\n⚠️ Warning: Dataset directory not found at {dataset_path}")
        print("You can still use the system by uploading your own documents")


def main():
    """Run setup"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     Medical RAG Chatbot - Installation Script             ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    # Step 1: Check Python version
    print("1️⃣ Checking Python version...")
    check_python_version()
    
    # Step 2: Create directories
    print("\n2️⃣ Creating directories...")
    create_directories()
    
    # Step 3: Install dependencies
    print("\n3️⃣ Installing dependencies...")
    response = input("Install dependencies now? (y/n): ").lower()
    if response == 'y':
        install_dependencies()
    else:
        print("⏭️ Skipped dependency installation")
        print("Run manually: pip install -r requirements.txt")
    
    # Step 4: Verify dataset
    print("\n4️⃣ Verifying dataset...")
    verify_dataset()
    
    # Done
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║  ✅ Installation Complete!                                ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print("\n📝 Next steps:")
    print("  1. Start server:      python server.py")
    print("  2. Open browser:      http://localhost:8000/docs")
    print("  3. Test system:       python test_system.py")
    print("\n💡 Tip: The first run will download models (~2-7GB)")
    print("    Make sure you have a good internet connection!\n")


if __name__ == "__main__":
    main()


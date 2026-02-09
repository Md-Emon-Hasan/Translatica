
import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add project root to python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    print(f"Starting Translatica UI & API...")
    print(f"Open http://localhost:8000 in your browser")
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

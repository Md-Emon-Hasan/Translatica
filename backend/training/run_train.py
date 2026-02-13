import sys
from pathlib import Path

# Add project root to python path so we can import 'training' module
# Get the directory containing this script (training/)
current_dir = Path(__file__).parent.absolute()
# Get the project root (parent of training/)
project_root = current_dir.parent.absolute()

sys.path.insert(0, str(project_root))

from training.train import main  # noqa: E402

if __name__ == "__main__":
    print("Starting Model Training...")
    # Run the main training function
    main()

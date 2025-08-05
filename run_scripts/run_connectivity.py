#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the package root to Python path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from reefconnect.scripts.get_connectivity import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_connectivity.py <release_start_day>")
        sys.exit(1)
    
    main(sys.argv[1]) 
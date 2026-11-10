"""
src/data/preprocess.py
----------------------
Backward-compatible proxy delegating to src/data/preprocessing.py.
"""

from src.data.preprocessing import preprocess, main

if __name__ == "__main__":
    main()

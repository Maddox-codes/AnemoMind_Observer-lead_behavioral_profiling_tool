import os
print("Current directory:", os.getcwd())
print("Files:", os.listdir(os.path.dirname(os.path.abspath(__file__))))     
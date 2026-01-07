import requests
import os

BASE_URL = "http://localhost:8000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    print(f"Root: {response.json()}")

def test_classify(file_path):
    print(f"Testing classification for: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f)}
        response = requests.post(f"{BASE_URL}/classify", files=files)
        if response.status_code == 200:
            print(f"Success: {response.json()}")
        else:
            print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Note: Ensure the server is running before executing this
    test_root()
    # Testing with an actual file from the dataset
    test_classify("dataset/business_1.txt")

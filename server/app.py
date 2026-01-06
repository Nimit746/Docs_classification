
from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Example GET endpoint
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# Example POST endpoint with request body
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return item

# Run with: uvicorn main:app --reload
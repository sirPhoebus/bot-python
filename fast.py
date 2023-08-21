from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

class User(BaseModel):
    username: str
    email: str

@app.get("/users/{user_id}")
async def read_user(user_id: int):
    # Retrieve user record by ID
    # ...
    # Return the user object
    return user

@app.post("/users/")
async def create_user(user: User):
    # Validate the input data
    if not user.username:
        raise HTTPException(status_code=400, detail="User name is required")
    if not user.email:
        raise HTTPException(status_code=400, detail="Email is required")

    # Create a new user record
    new_user = User(username=user.username, email=user.email)

    # Save the new user record to the database
    print(new_user)

    # Return the new user object
    return new_user

if __name__ == "__main__":
  
    uvicorn.run(app, host="localhost", port=8000)

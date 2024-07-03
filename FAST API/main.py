from fastapi import FastAPI, Path, Query, Depends, BackgroundTasks, File, UploadFile
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()

# Simple Exercises

# 1. Hello World Endpoint
@app.get("/hello")
def hello_world():
    """Returns 'Hello, World!' message."""
    return {"message": "Hello, World!"}

class User(BaseModel):
    name: str
    age: int
    email: EmailStr

@app.post("/user")
def create_user(user: User):
    return {"message": f"User {user.name} created successfully!"}


# 2. Path Parameters
@app.get("/hello/{name}")
def greet_name(name: str = Path(..., description="Name to greet")):
    """Returns 'Hello, {name}!'"""
    return {"message": f"Hello, {name}!"}

# 3. Query Parameters
@app.get("/greet")
def greet_query(first_name: str = Query(..., description="First name"), last_name: str = Query(..., description="Last name")):
    """Returns 'Hello, {first_name} {last_name}!'"""
    return {"message": f"Hello, {first_name} {last_name}!"}

# Medium Exercises

# 1. Data Validation with Pydantic
class User(BaseModel):
    name: str
    age: int
    email: EmailStr

@app.post("/user")
def create_user(user: User):
    """Creates a user and returns a success message if data is valid."""
    return {"message": f"User {user.name} created successfully!"}

# 2. CRUD Operations
items = []

@app.get("/items")
def read_items():
    """Returns all items."""
    return items

@app.post("/items")
def create_item(item: dict):
    """Adds a new item to the list."""
    items.append(item)
    return item

@app.put("/items/{id}")
def update_item(id: int, item: dict):
    """Updates an item by id."""
    if id < len(items):
        items[id] = item
        return item
    return {"error": "Item not found"}

@app.delete("/items/{id}")
def delete_item(id: int):
    """Deletes an item by id."""
    if id < len(items):
        return items.pop(id)
    return {"error": "Item not found"}

# 3. Dependency Injection
class DatabaseConnection:
    def __init__(self):
        self.connection = "Database Connection"

def get_db():
    db = DatabaseConnection()
    try:
        yield db
    finally:
        pass

@app.get("/db_items")
def read_items_from_db(db: DatabaseConnection = Depends(get_db)):
    """Fetches items using a database connection dependency."""
    return {"db_connection": db.connection, "items": items}

# Complex Exercises

# 1. User Authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Returns a token for user authentication."""
    return {"access_token": form_data.username, "token_type": "bearer"}

@app.get("/users/me")
def read_users_me(token: str = Depends(oauth2_scheme)):
    """Fetches the current user, protected by OAuth2 authentication."""
    return {"user": token}

# 2. File Upload and Download
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """Accepts file uploads and saves them to the server."""
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@app.get("/download/{filename}")
def download_file(filename: str):
    """Allows users to download the uploaded files."""
    file_location = f"files/{filename}"
    return FileResponse(file_location)

# 3. Background Tasks
def background_task(name: str):
    # Simulate a long-running process
    import time
    time.sleep(10)
    print(f"Processed data for {name}")

@app.post("/process")
def process_data(background_tasks: BackgroundTasks, name: str):
    """Triggers a background task to process data and returns 'Processing started'."""
    background_tasks.add_task(background_task, name)
    return {"message": "Processing started"}

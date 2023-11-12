import sys
import os

# Get the current script's directory
current_script_directory = os.path.dirname(os.path.abspath(__file__))

# Get the project root path
project_root = os.path.abspath(os.path.join(current_script_directory, os.pardir))

sys.path.append(project_root)

from typing import List, Optional

from fastapi import Depends, FastAPI, Response
from scripts.txt2img import txt2img
from starlette.responses import RedirectResponse
from starlette.status import HTTP_201_CREATED

from api.repository import Todo, TodoFilter, TodoRepository, create_todo_repository

app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})


@app.get("/")
async def root():
    return RedirectResponse(app.docs_url)


@app.post("/txt2img", status_code=HTTP_201_CREATED)
async def t2i(prompt: str):
    result = txt2img(prompt)
    result.save("output.png")

    # Return the image file as the response content
    with open("output.png", "rb") as f:
        file_content = f.read()

    return Response(content=file_content, media_type="image/png")


@app.post("/create/{key}", status_code=HTTP_201_CREATED)
def create(key: str, value: str, todo_repository: TodoRepository = Depends(create_todo_repository)):
    with todo_repository as repo:
        repo.save(Todo(key=key, value=value))


@app.get("/get/{key}", response_model=Optional[Todo])
def get(key: str, todo_repository: TodoRepository = Depends(create_todo_repository)):
    with todo_repository as repo:
        return repo.get_by_key(key)


@app.get("/find", response_model=List[Todo])
def find(todo_filter: TodoFilter = Depends(), todo_repository: TodoRepository = Depends(create_todo_repository)):
    with todo_repository as repo:
        return repo.get(todo_filter)

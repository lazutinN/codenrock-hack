import fastapi
import os

from fastapi import APIRouter, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

from pathlib import Path

app=fastapi.FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "rest/static"),
    name="static",
)
templates = Jinja2Templates(directory="templates")

@app.get('/', status_code=200)
def hellopage(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
        )

@app.post('/upload_file/{file_name}')
def uploadFile(file: str):
    return file
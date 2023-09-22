import fastapi
import os

from fastapi import APIRouter, Request, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

from pathlib import Path

app=fastapi.FastAPI()

#обозначаем где находится папка static
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "rest/static"),
    name="static",
)
templates = Jinja2Templates(directory="templates")

@app.get('/', status_code=200)
# ручка главной страницы, по-дефолту (при первом визите) возвращает index.html и styles.css
#
#
def hellopage(request: Request):
        return templates.TemplateResponse(
            "index.html", {"request": request}
            )

def uploadFile(file: UploadFile(...)):
     file_location = f"rest/data/downloaded/{file.filename}"

     with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())    
        return RedirectResponse(url="/status", status_code=HTTP_303_SEE_OTHER)



@app.get('/status', status_code=200)
def status(request: Request):
    return templates.TemplateResponse(
        "status.html", {"request": request}
    )
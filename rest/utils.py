import os
import datetime

from pydantic import *


class File(BaseModel):
    id = int
    fileName = str
    uploadTime = datetime.datetime

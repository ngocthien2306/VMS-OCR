import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from utils.socket import socket_connection
from utils.project_config import project_config
from router.docs_router import add_docs_router
from router import event_router
from router import history_router
from router import report_router
from router import camera_router
from router import server_router

app = FastAPI(
    version=1.0,
    title=project_config.DOCS_TITLE,
    debug=project_config.DEBUG, 
    docs_url=None, 
    redoc_url=None
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/ws", socket_connection())
app.mount("/public", StaticFiles(directory="public"), name="public")


add_docs_router(app)
app.include_router(event_router.router, tags=["Event"])
app.include_router(history_router.router, tags=["History"])
app.include_router(report_router.router, tags=["Report"])
app.include_router(camera_router.router, tags=["Camera"])
app.include_router(server_router.router, tags=["Server"])

if __name__ == '__main__':
    uvicorn.run(app, host="172.20.10.2", port=project_config.BE_PORT)

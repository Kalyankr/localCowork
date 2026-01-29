import uvicorn

from agent.config import settings
from agent.orchestrator.server import app

if __name__ == "__main__":
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)

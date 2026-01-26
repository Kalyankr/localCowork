import uvicorn
from agent.orchestrator.server import app
from agent.config import settings

if __name__ == "__main__":
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)

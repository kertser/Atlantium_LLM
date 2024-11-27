import uvicorn
from config import CONFIG

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=9000,
        reload=True
    )
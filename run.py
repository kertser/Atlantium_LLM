import uvicorn
from config import CONFIG

if __name__ == "__main__":


    print("Starting server...")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=CONFIG.SERVER_PORT,
        reload=True
    )
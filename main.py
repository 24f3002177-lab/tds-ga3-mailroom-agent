from fastapi import FastAPI

app = FastAPI(title="Mailroom Agent")


@app.get("/")
def home():
    return {"message": "Mailroom Agent", "status": "running"}

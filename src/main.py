from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def main():
    return {"Message": "API is running"}
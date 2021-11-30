import uvicorn 
from fastapi import FastAPI

app = FastAPI()

#Create Routes
@app.get('/api')
async def index():
    return {'text':"Changes made"}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.0', port=8000)
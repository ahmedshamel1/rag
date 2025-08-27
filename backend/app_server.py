from fastapi import FastAPI
import uvicorn

from chat_agents import multi_role_agent
from chat_agents import hr_agent

app = FastAPI()

@app.on_event("startup")
def startup_event():
    # initialize agents once
    try:
       # hr_agent.init_hr_agent()
        # if you have multi-role graph, init it too
        # multi_role_agent.init_multi_role_agent()
        print("Agents initialized on startup")
    except Exception as e:
        print("Failed to init agents:", e)

@app.on_event("shutdown")
def shutdown_event():
    # persist stores if necessary; e.g. chroma persist handled automatically in many clients,
    # but if you need explicit flush do it here.
    print("Shutting down - persist / cleanup if needed")

@app.get("/")
async def root():
    return {"message": "Welcome to Multi-Agent Virtual Assistant"}

from pydantic import BaseModel

class ProcessRequestBody(BaseModel):
    user_prompt: str
    agent: str

@app.post("/process/")
async def process_prompt(request_data: ProcessRequestBody):
    user_prompt = request_data.user_prompt
    agent = request_data.agent
    if agent == "bakers":
        return {"response": multi_role_agent.get_baker_response(user_prompt)}
    elif agent == "hr":
        return {"response": hr_agent.hr_qa(user_prompt)}
    elif agent == "cofounder":
        return {"response": multi_role_agent.get_cofounder_response(user_prompt)}
    else:
        return {"response": "Unknown agent specified. Use 'bakers', 'hr', or 'cofounder'."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

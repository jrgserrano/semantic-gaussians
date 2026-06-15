from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
import json

app = FastAPI(title="Semantic Navigator LLM Agent")

# allow CORS so the WebGL frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional

class LabelItem(BaseModel):
    name: str
    description: str

class ReasonRequest(BaseModel):
    query: str
    available_labels: list[LabelItem]
    image: Optional[str] = None

llm = ChatOllama(model="nemotron-3-nano", temperature=0.0, num_ctx=16384, format="json")

system_template = """You are a highly intelligent navigation assistant for a robot.
Your goal is to map the user's abstract query to EXACTLY ONE of the available object labels in the room.

IMPORTANT RULES:
1. The user's query may be in Spanish or English. Translate the intent internally before matching.

2. Use the 'Description' of each object to make your decision. Do NOT rely solely on the 'Name'. The description contains the actual semantic context.

Respond ONLY with a valid JSON object in this exact format. 
EXAMPLE OUTPUT:
{{
  "reasoning": "step-by-step reasoning explaining why this object was selected based on the user query",
  "target": "the exact 'name' of the selected label"
}}

Do not include any explanations outside the JSON.
"""

human_template = """Available labels are provided in the following format:
- Name: 'label_name' | Description: A detailed description of the object.

{labels}

User's query: "{query}"

Analyze the user's intent and select the single best matching label from the list above. The description contains the actual semantic context.

Answer:"""


chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

@app.post("/reason")
async def reason_target(request: ReasonRequest):
    if not request.available_labels:
        raise HTTPException(status_code=400, detail="No labels provided.")
    
    labels_str = "\n".join([f"- Name: '{lbl.name}' | Description: {lbl.description}" for lbl in request.available_labels])
    text_prompt = chat_prompt.format_messages(query=request.query, labels=labels_str)[1].content
    human_msg = HumanMessage(content=text_prompt)
        
    sys_msg = SystemMessagePromptTemplate.from_template(system_template).format()
    messages = [sys_msg, human_msg]
    
    try:
        print(f"[*] Query received: '{request.query}'")
        # run inference
        response_msg = llm.invoke(messages)
        response_text = response_msg.content.strip()
        print(f"[*] LLM Output: {response_text}")
        
        try:
            parsed = json.loads(response_text)
            target = str(parsed.get("target", parsed.get("label", parsed.get("name", ""))))
            reasoning = str(parsed.get("reasoning", ""))
        except json.JSONDecodeError:
            print("[Warning] Could not parse JSON, falling back to raw text")
            target = response_text.replace('"', '').replace("'", "").strip()
            reasoning = ""
            
        if reasoning:
            print(f"[*] Reasoning: {reasoning}")
        return {"target": target, "reasoning": reasoning}
    except Exception as e:
        print(f"[Error] LLM Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from groq import Groq

# Set up your API key and base URL
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")
client = Groq(api_key=API_KEY)

# Initialize FastAPI
app = FastAPI()

class QueryRequest(BaseModel):
    exercises: str
    target: str
    gender: str
    cuisine: str
    allergies: str

class QueryResponse(BaseModel):
    response: str

def generate_response(query, system_prompt, model="llama-3.1-70b-versatile"):
    """Generate a response using the Groq API."""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        model=model,
        stream=False,
        temperature=0.3,
        max_tokens=7000,
    )
    return response.choices[0].message.content

# Hardcoded system prompt
SYSTEM_PROMPT = """1.) Your only role is to provide people a diet plan for 1 week
2.) You will be given the workout routine and the target which a person is trying to achieve along with it with the type of cuisine they prefer
3.) Understand the workout plan they follow along with the target they plan to achieve
4.) After COMPLETELY understanding the input, you have to generate a 7 day meal plan consisting of breakfast, lunch and dinner.
5.) MOST IMPORTANT: Your only job is to UNDERSTAND completely and generate a proper 7 day meal plan based on the TARGET, Cuisine, and Allergies 
6.) If you see a "+" under target it means they are trying to put on weight, but if you see a "-" it means they are trying to lose weight
7.) You should clearly mention the quantity of the food which is supposed to be taken CLEARLY ALWAYS along with the calories and protein present!!!!
"""

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        # Formulate the query sentence
        query_sentence = (
            f"Give a 7 day meal plan under the {request.cuisine} cuisine for a {request.gender}, "
            f"who follows these exercises {request.exercises} to reach the target of {request.target}. "
            f"Allergies to this person are {request.allergies}."
        )

        # Print the formed sentence for debugging
        print(f"Formed Query Sentence: {query_sentence}")

        # Generate a response
        response = generate_response(query_sentence, SYSTEM_PROMPT)

        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

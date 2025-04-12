from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
import os
import json
import re
from datetime import datetime
import uuid
from dotenv import load_dotenv
import openai
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Fitness Challenge Generator API",
    description="API for generating personalized fitness plans",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define input model
class FitnessGoalInput(BaseModel):
    age: int = Field(..., ge=16, le=80, description="User age in years")
    height: float = Field(..., ge=120, le=220, description="User height in centimeters")
    weight: float = Field(..., ge=30, le=200, description="User weight in kilograms")
    gender: Optional[str] = Field(None, description="User gender (optional)")
    goal: str = Field(..., min_length=5, max_length=200, description="Fitness goal description")
    timeline: int = Field(..., ge=1, le=12, description="Goal timeline in months")
    fitness_level: Optional[str] = Field(None, description="Current fitness level (beginner, intermediate, advanced)")
    limitations: Optional[str] = Field(None, description="Any physical limitations or health concerns")
    
    @validator('goal')
    def goal_must_be_valid(cls, v):
        blacklist = ['illegal', 'steroids', 'drugs', 'harm', 'abuse', 'suicide']
        if any(keyword in v.lower() for keyword in blacklist):
            raise ValueError("Goal contains inappropriate or harmful content")
        return v

# Define response model
class FitnessPlanResponse(BaseModel):
    plan_id: str
    created_at: str
    summary: str
    daily_routines: Dict[str, Any]
    nutrition_plan: Dict[str, Any]
    progress_tracking: Dict[str, Any]

def parse_json_safely(text: str) -> dict:
    """
    Attempt to parse JSON from text using multiple strategies
    """
    parsing_strategies = [
        # 1. Direct JSON parsing
        lambda t: json.loads(t),
        
        # 2. Extract JSON between ```json and ```
        lambda t: json.loads(re.search(r'```json\s*({.*?})\s*```', t, re.DOTALL).group(1)),
        
        # 3. Extract JSON between ``` and ```
        lambda t: json.loads(re.search(r'```\s*({.*?})\s*```', t, re.DOTALL).group(1)),
        
        # 4. Find first JSON-like object
        lambda t: json.loads(re.search(r'\{.*\}', t, re.DOTALL).group(0))
    ]
    
    for strategy in parsing_strategies:
        try:
            return strategy(text)
        except Exception as e:
            logger.debug(f"Parsing strategy failed: {str(e)}")
    
    # If all strategies fail, raise an exception
    raise ValueError("No valid JSON found in the response")

async def generate_fitness_plan(user_input: FitnessGoalInput):
    """Generate a personalized fitness plan using LLM"""
    
    # Construct the prompt for the LLM
    prompt = f"""
    Create a detailed {user_input.timeline}-month fitness plan for a {user_input.age}-year-old 
    {'person' if not user_input.gender else user_input.gender} who is {user_input.height}cm tall, 
    weighs {user_input.weight}kg, and wants to achieve: {user_input.goal}.
    
    {'Their current fitness level is: ' + user_input.fitness_level if user_input.fitness_level else ''}
    {'They have the following limitations: ' + user_input.limitations if user_input.limitations else ''}
    
    IMPORTANT: Respond ONLY with a valid JSON object in the following exact format:
    {{
      "summary": "Overall plan summary and approach (string)",
      "daily_routines": {{
        "week1": {{
          "monday": {{
            "workout": ["Exercise 1: 3 sets x 10 reps", "Exercise 2: 3 sets x 12 reps"],
            "cardio": "30 minutes moderate intensity running",
            "rest": "Light stretching, hydration"
          }},
          "tuesday": {{ ... }}
        }},
        "week2": {{ ... }}
      }},
      "nutrition_plan": {{
        "daily_calories": 2200,
        "macros": {{
          "protein": 180,
          "carbs": 250,
          "fats": 65
        }},
        "meal_plan": {{
          "breakfast": ["Protein smoothie", "Oatmeal with berries"],
          "lunch": ["Grilled chicken salad", "Quinoa bowl"],
          "dinner": ["Salmon with roasted vegetables", "Lean beef stir-fry"],
          "snacks": ["Greek yogurt", "Protein bar", "Almonds"]
        }},
        "hydration": "3 liters of water daily"
      }},
      "progress_tracking": {{
        "metrics_to_track": ["Body weight", "Body fat percentage", "Muscle measurements"],
        "milestone_expectations": {{
          "month1": "Lose 4-5 pounds, increase strength by 10%",
          "month2": "Lose additional 4-5 pounds, improve cardiovascular endurance"
        }}
      }}
    }}
    
    ENSURE:
    - Strictly use the JSON format above
    - Include realistic, achievable goals
    - Provide specific, actionable advice
    - Tailor the plan to the user's specific goal and fitness level
    """
    
    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Or other model of your choice
            messages=[
                {"role": "system", "content": "You are a professional fitness coach and nutritionist. Create personalized fitness plans as a VALID JSON object. ONLY respond with the JSON, no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        # Parse JSON safely
        try:
            plan_data = parse_json_safely(response_text)
        except Exception as json_error:
            # Log full response for debugging
            logger.error(f"JSON Parsing Error: {json_error}")
            logger.debug(f"Full Response Text: {response_text}")
            raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {str(json_error)}")
        
        # Add metadata
        plan_data["plan_id"] = str(uuid.uuid4())
        plan_data["created_at"] = datetime.now().isoformat()
        
        return plan_data
        
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate fitness plan: {str(e)}")

# Define endpoints
@app.post("/generate-plan/", response_model=Dict[str, Any], tags=["Fitness Plans"])
async def create_fitness_plan(user_input: FitnessGoalInput):
    """
    Generate a personalized fitness plan based on user input.
    
    This endpoint accepts user physical stats, fitness goals, and timeline,
    then generates a detailed daily fitness and nutrition plan.
    """
    plan = await generate_fitness_plan(user_input)
    return plan

@app.get("/health", tags=["System"])
async def health_check():
    """Endpoint to check if API is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Add an endpoint to serve API documentation
@app.get("/", tags=["Documentation"])
async def root():
    """Redirect to API documentation"""
    return {"message": "Welcome to Fitness Challenge Generator API. Visit /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
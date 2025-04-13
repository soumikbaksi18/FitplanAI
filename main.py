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
    title="Location-Based Fitness Challenge Generator API",
    description="API for generating personalized fitness plans based on location",
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
    location: Optional[str] = Field(..., min_length=2, max_length=100, description="User's location to customize diet (e.g., USA, London, Delhi, Kolkata, Bangalore, Chennai)")
    
    @validator('goal')
    def goal_must_be_valid(cls, v):
        blacklist = ['illegal', 'steroids', 'drugs', 'harm', 'abuse', 'suicide']
        if any(keyword in v.lower() for keyword in blacklist):
            raise ValueError("Goal contains inappropriate or harmful content")
        return v
    
    @validator('location')
    def location_must_be_valid(cls, v):
        # Optional: Add validation for supported locations if needed
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
    
    # Create a JSON template string separately to avoid nested f-string issues
    json_template = '''
    {
      "summary": "Overall plan summary and approach (string)",
      "daily_routines": {
        "week1": {
          "monday": {
            "workout": ["Exercise 1: 3 sets x 10 reps", "Exercise 2: 3 sets x 12 reps"],
            "cardio": "30 minutes moderate intensity running",
            "rest": "Light stretching, hydration"
          },
          "tuesday": { ... }
        },
        "week2": { ... }
      },
      "nutrition_plan": {
        "daily_calories": 2200,
        "macros": {
          "protein": 180, 
          "carbs": 250,
          "fats": 65
        },
        "meal_plan": {
          "breakfast": ["High protein option 1", "High protein option 2"],
          "lunch": ["Lean protein with vegetables 1", "Lean protein with vegetables 2"],
          "dinner": ["Protein-rich dinner 1", "Protein-rich dinner 2"],
          "snacks": ["Healthy protein snack 1", "Healthy protein snack 2"]
        },
        "hydration": "3 liters of water daily",
        "regional_foods": ["List of location-specific HEALTHY food recommendations"]
      },
      "progress_tracking": {
        "metrics_to_track": ["Body weight", "Body fat percentage", "Waist measurement"],
        "milestone_expectations": {
          "month1": "Lose 1-2kg of fat, increase core strength by 10%",
          "month2": "Continue fat loss, improve muscle definition"
        }
      }
    }
    '''
    
    # Construct the prompt for the LLM without nesting the full JSON template in the f-string
    prompt = f"""
    Create a detailed {user_input.timeline}-month fitness plan for a {user_input.age}-year-old 
    {'person' if not user_input.gender else user_input.gender} who is {user_input.height}cm tall, 
    weighs {user_input.weight}kg, and wants to achieve: {user_input.goal}.
    
    {'Their current fitness level is: ' + user_input.fitness_level if user_input.fitness_level else ''}
    {'They have the following limitations: ' + user_input.limitations if user_input.limitations else ''}
    
    IMPORTANTLY, they are located in: {user_input.location}
    Create a nutrition plan specifically tailored to foods and dishes commonly available in {user_input.location},
    but ensure all suggested foods are OPTIMIZED FOR THE GOAL of {user_input.goal}.
    
    The nutrition plan MUST include:
    1. High protein foods to support muscle building
    2. Limited carbohydrates to reduce body fat and reveal abdominal muscles
    3. Moderate healthy fats
    4. Focus on whole, unprocessed foods
    5. Appropriate portion sizes
    
    For each location, provide HEALTHY versions or alternatives of local cuisine:
    - If in USA: Lean proteins (turkey, chicken breast), whole grains, vegetables, etc.
    - If in Japan: Fish, tofu, seaweed, edamame, miso soup (low sodium), etc.
    - If in Delhi (North India): Tandoori chicken (skinless), dal (lentils), paneer (in moderation), roti (whole grain, limited quantity), etc.
    - If in Kolkata (East India): Baked/steamed fish (instead of fried), vegetable curry (low oil), dal, brown rice (limited portion), etc.
    - If in Bangalore/Chennai (South India): Egg whites in dosa, vegetable upma, rasam, sambhar (low oil), etc.
    
    IMPORTANT: Avoid or minimize sugary desserts, fried foods, and high-carb items in the meal plan. Modify traditional dishes to be healthier versions.
    
    IMPORTANT: Respond ONLY with a valid JSON object in the following exact format:
    {json_template}
    
    ENSURE:
    - Strictly use the JSON format above
    - Include realistic, achievable goals
    - Provide specific, actionable advice
    - Tailor the plan to the user's specific goal, fitness level, and LOCATION
    - The nutrition plan should feature HEALTHY foods commonly available in their location
    - ALL food recommendations must support the goal of {user_input.goal}
    """
    
    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Or other model of your choice
            messages=[
                {"role": "system", "content": "You are a professional fitness coach and nutritionist with expertise in global cuisine AND sports nutrition. Create personalized fitness plans as a VALID JSON object with location-specific HEALTHY food recommendations that support the user's fitness goals. ONLY respond with the JSON, no additional text."},
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
        plan_data["location"] = user_input.location  # Include location in the response
        
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
    Generate a personalized fitness plan based on user input and location.
    
    This endpoint accepts user physical stats, fitness goals, timeline, and location,
    then generates a detailed daily fitness and nutrition plan with region-specific food recommendations.
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
    return {"message": "Welcome to Location-Based Fitness Challenge Generator API. Visit /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict

app = FastAPI(
    title="Wine Classifier API",
    description="ML model for wine classification",
    version="1.0.0"
)

class Features(BaseModel):
    """Request model for prediction"""
    features: list[float]

class PredictionResponse(BaseModel):
    """Response model"""
    class_id: int
    class_name: str
    probabilities: list[float]

@app.get("/")
def read_root():
    return {"message": "Wine Classifier API is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def make_prediction(data: Features):
    """Make a prediction on input features"""
    try:
        # Validate feature count (Wine dataset has 13 features)
        if len(data.features) != 13:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 13 features, got {len(data.features)}"
            )
        
        result = predict(data.features)
        
        return PredictionResponse(
            class_id=result['class'],
            class_name=result['class_name'],
            probabilities=result['probabilities']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def make_batch_predictions(data: list[Features]):
    """Make predictions on multiple samples"""
    try:
        results = []
        for item in data:
            result = predict(item.features)
            results.append({
                'class_id': result['class'],
                'class_name': result['class_name'],
                'probabilities': result['probabilities']
            })
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
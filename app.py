from fastapi import FastAPI
from pydantic import BaseModel
from semantic_engine import SemanticEngine

app =  FastAPI() #defines a new FastAPI application
engine = SemanticEngine(data_path="")

class SearchRequest(BaseModel):
    query: str
    employee_id: str
    top_k: int = 5
    threshold: float = 0.5
    noreg: str

@app.post("/search") # defines a post endpoint at /search
def search(request: SearchRequest): #function that runs when someone visits /search
        results = engine.search(request.employee_id, 
                                request.query, 
                                request.top_k, 
                                request.threshold) # returns a JSON to the client
        
        if 'concept' in results.keys():
             labcheck_results = engine.labcheck_search(request.employee_id,
                                                     concept_checkup_cd=results['concept_checkup_cd'])
             hospi_results = engine.hospi_search(request.noreg,
                                                 concept = results['concept'],
                                                 concept_icd = results['concept_icd']) # i want hospi to be independent
        else:
             labcheck_results = {"status":"not_found"}
             hospi_results = {"status":"not_found"}

        return {"results": results,
                "labcheck": labcheck_results,
                "hospitalization_result": hospi_results}

# uvicorn app:app --reload --port 8001 

    



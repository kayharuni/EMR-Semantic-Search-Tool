# EMR-Semantic-Search-Tool
This project aims to create a semantic search engine that allows users in this case doctors or clinicians to retrieve medical notes or phrases from EMR that are semantically similar or relevant to the queries/keywords. 

## Name
Semantic Search Tool for EMR (Employee Medical Records)

## Description
This project/service allows user or clinicians to enter an input or medical query and retrieve medical notes relevant to that query based on semantic similarity. Additionally, the engine has a keyword-based filtering system to identify lab check up abnormalities and hospitalization records relevant to said query. The purpose of this project is to automate and make efficient the assessment process of EMR as well as to identify risks. 


## Installation
Python 3.10.19\
install sentence-transformers 5.2.3\
install fastapi 0.129.0\
install pydantic 2.12.5

## Usage

**INPUT PARAMETERS**\
query = "diabetes" --> medical keyword or phrase to retrieve medical notes/records, better if it mentions the disease name\
employee_id = "P004" --> employee identifier to filter data\
top_k = 10 --> top k most similar ranking to the query\
threshold = 0.5 --> threshold cutoff for minimum similarity score to filter relevance \
noreg = "0012344" --> registration number to filter  for hospitalization records\

**OUTPUT**\
(1) Semantic search of medical notes\
\
"source": "anamnesa",\
"feature": "CURR_MEDICATION_MEDICINE",\
"transaction_date": "2023-12-26",\
"note": "Insulin Sejak 2 Tahun Terakhir",\
"similarity_score": 0.84\
\
(2) Abnormal findings of lab check up\
\
"LAB_ID": "1833959",\
"TRANS_NO": "RS-0001234073923",\
"CHECKUP_ID": "987",\
"LAB_VALUE": "170",\
"CONDITION": "ABNORMAL",\
"CHECKUP_NAME": "Glukosa Puasa",\
"CHECKUP_CD": "CHK-123"\
\
(3) Hospitalization result\
\
NOREG": "00123482",\
"ADMISSION_DATE": "2018-06-07 00:00:00.000",\
"PRIMARY_DESC": "NON-INSULIN-DEPENDENT DIABETES MELLITUS",\
"ICD": "E11"


## Direction

semantic_engine.py --> Semantic search engine pipeline responsible for embedding generation, similarity computation, and medical concept detection.\
app.py --> FastAPI backend API server that exposes the semantic search engine through REST endpoints.


## Support
kaylaharuni@gmail.com



## Project status
Development has been handed over.

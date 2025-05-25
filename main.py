from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import csv

app = FastAPI()

# Enable CORS (allow all origins for GET)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"]
)

# Read the CSV file once at startups
students_data = []
with open("q-fastapi.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for i, row in enumerate(reader, start=1):
        try:
            student_id = int(row["studentId"])
            class_name = row["class"]
            students_data.append({
                "studentId": student_id,
                "class": class_name
            })
        except Exception as e:
            print(f"Skipping row {i}: {row}, error: {e}")


@app.get("/api")
def get_students(class_: Optional[List[str]] = Query(None, alias="class")):
    
    if class_:
        filtered = [s for s in students_data if s["class"] in class_]
    else:
        filtered = students_data
    return {"students": filtered}

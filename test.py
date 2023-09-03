from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uuid

app = FastAPI()

class School(BaseModel):
    name: str

class Student(BaseModel):
    id: str
    name: str
    age: int
    school: School

students_by_school = {}

@app.get("/schools", response_model=List[School])
def get_all_schools():
    return [School(name=school) for school in students_by_school.keys()]

@app.post("/schools/{school_name}", response_model=Student)
def create_student(school_name: str, student: Student):
    if school_name not in students_by_school:
        students_by_school[school_name] = []
    
    new_student = Student(id=str(uuid.uuid4()), **student.dict(), school={"name": school_name})
    students_by_school[school_name].append(new_student)
    return new_student

@app.delete("/schools/{school_name}/students/{student_id}")
def delete_student(school_name: str, student_id: str):
    try:
        students_by_school[school_name].remove([s for s in students_by_school[school_name] if s.id == student_id][0])
    except (IndexError, ValueError):
        raise HTTPException(status_code=404, detail="Student not found")

@app.put("/schools/{school_name}/students/{student_id}")
def update_student(school_name: str, student_id: str, student: Student):
    try:
        index = next((i for i, s in enumerate(students_by_school[school_name]) if s.id == student_id))
        updated_student = Student(id=student_id, **student.dict())
        students_by_school[school_name][index] = updated_student
        return updated_student
    except (StopIteration, KeyError):
        raise HTTPException(status_code=404, detail="Student not found")

@app.get("/schools/{school_name}/students", response_model=List[Student])
def get_students_in_school(school_name: str) -> List[Student]:
    try:
        return [Student(id=s.id, **s.dict()) for s in students_by_school[school_name]]
    except KeyError:
        raise HTTPException(status_code=404, detail="School not found")

@app.get("/schools/{school_name}/students/{student_id}", response_model=Optional[Student])
def get_student_details(school_name: str, student_id: str) -> Optional[Student]:
    try:
        return Student(id=student_id, **next((s for s in students_by_school[school_name] if s.id == student_id)).dict())
    except (StopIteration, KeyError):
        raise HTTPException(status_code=404, detail="School or student not found") 
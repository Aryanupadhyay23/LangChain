from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Aryan"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0.0, lt=10.0, default = 5)

new_student = {"age":"21", "email": "john@example.com"}
student = Student(**new_student)
student_dict = dict(student)
print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)
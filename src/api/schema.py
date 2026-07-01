from pydantic import BaseModel


class StudentData(BaseModel):
    Age: int
    Gender: int
    Ethnicity: int
    ParentalEducation: int
    StudyTimeWeekly: float
    Absences: int
    Tutoring: int
    ParentalSupport: int
    Extracurricular: int
    Sports: int
    Music: int
    Volunteering: int
    GPA: float
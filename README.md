# Diploma
Emotion recognition
graph TD

U[User] --> F[Frontend (React / Next.js)]
F --> B[Backend API (FastAPI)]

B --> A[Audio Feature Extraction]
B --> T[Text Processing (Whisper + BERT)]
B --> V[Visual Processing (OpenFace / PyFeat)]

A --> M[BottleneckFusionModel]
T --> M
V --> M

M --> B
B --> F

F --> R[Display Emotion Result]
```mermaid
%%{init: {'theme':'base', 'themeVariables': {
  'background': '#ffffff',
  'primaryColor': '#ffffff',
  'primaryBorderColor': '#000000',
  'lineColor': '#000000',
  'textColor': '#000000'
}}}%%

graph TD

U[User] --> F[Frontend React / Next.js]
F --> B[Backend API FastAPI]

B --> P[Feature Extraction Pipeline]

P --> A[Audio FFmpeg + OpenSMILE]
P --> T[Text Whisper + BERT]
P --> V[Visual OpenFace / PyFeat]

A --> M[BottleneckFusionModel]
T --> M
V --> M

M --> O[Emotion Prediction]

O --> B
B --> F

F --> R[Result Display]
```
Project Title: GonaX - Blood Cockle Classifier.

Description: Currently, gonadal stage determination relies on manual histological examination, which is time-consuming, subjective, and dependent on expert interpretation. This creates possible inconsistencies in classification, especially for researchers in the UPV Institute of Aquaculture who have been conducting histological staging manually.  Because of this, our motivation is to create a system that address these problems by making classification faster, more objective, and reproducible.

Tech Stack: 
    Model Training:
    Python, Scikit, Opencv, Numpy, Pandas, Pickle, Joblib, Tensorflow.

    Web Application:
    Typescript, Next.Js, Tailwind. fastapi, uvicorn, pydantic

Dependencies or software packages required before running the program.

fastapi
uvicorn
numpy
opencv-python-headless
pydantic
xgboost
imbalanced-learn
pillow
pandas
python-multipart
seaborn
scikit-image
scikit-learn
scipy
torch
torchvision
torchaudio

Installation: Step-by-step terminal commands or setup workflows needed to build locally.
1. Go to the backend directory and create a virtual environment
    cd gonad-app/backend/
    python -m venv .venv
2. Activate virtual environment and Install the necessary dependencies from the requirements.txt
    venv\scripts\activate
    pip install -r requirements.txt
3. Create a new terminal and run the local server
    npm run dev
4. Create a new terminal and find the gonad-app/backend directory and run uvicorn
    uvicorn main:app --reload
5. Open the web application and upload your histological images works best with High definition images  
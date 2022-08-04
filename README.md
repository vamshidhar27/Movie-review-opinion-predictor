# Movie review opinion predictor
To run this project first download and extract it 
## Install Python 
To check if python is installed or not type the following in command prompt
```
python --version
```
At the time of building this project the version of python that I was using was 3.9.7
## Install Virtual Environment
```
pip install virtualenv
```
## Creating Virtual Environment 
Navigate to the downloaded project using cd command and specifing its path 
```
cd path
```
Running the below command creates a virtual environment
```
python -m venv venv
```
## Activating Virtual Environment
```
venv\scripts\activate
```
## Installing Dependencies
```
pip install -r requirements.txt
```
## Downloading the Dataset
Download the dataset in the project directory <br> <br>
https://ai.stanford.edu/~amaas/data/sentiment/
## Exceuting the Project 
Go to the project directory using cd 
```
cd path
```
Run the app.py file <br>
```
python app.py
```
First train the model once <br>
Training the model creates a pickle file named my_model <br>
Now again run app.py file <br>
```
python app.py
```
Now test the custom review <br>
A link is generated using which you can test the review <br>



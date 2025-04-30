# Big Data – Logistic Regression with Iris Dataset

## Uploaded Files

- `requirements.txt` – Python dependencies used in the project  
- `Dockerfile` – Docker configuration to containerize the application  
- `2task_bd.py` – Main Python script with Flask app and logistic regression logic  
- DockerHub Repository: https://hub.docker.com/repository/docker/ugnekniukstaite/2-task-big-data

---

##  requirements.txt

This file lists the Python libraries required to run the application:

Flask==3.1.0  
pandas==1.5.1  
plotly==6.0.1  
scikit-learn==1.3.2

These libraries are used for data handling, visualization, machine learning, and building the web app. The only libraries, what I used in my python file.

---

##  Dockerfile 

FROM python:3.9.13-slim-buster  
→ Uses a lightweight Python 3.9 image based on Debian Buster.

WORKDIR /app  
→ Sets the working directory inside the container to `/app`.

COPY 2task_bd.py ./  
COPY requirements.txt ./  
→ Copies the application code and dependency list into the container.

RUN pip install -r requirements.txt  
→ Installs the required Python libraries.

EXPOSE 5000  
→ Tells Docker to expose port 5000 for the Flask web server.

CMD ["python", "2task_bd.py"]  
→ Sets the default command to run the Flask app when the container starts.

---

## 2task_bd.py – Project Overview

This Flask-based Big Data project performs logistic regression on the classic Iris dataset. The web interface provides multiple visualizations and model performance metrics.

### Key Features

- Dataset: Built-in Iris dataset from Plotly, with flower species and measurements.
- Model: Logistic Regression from scikit-learn, trained on 70% of the data.
- Visualizations:
  - ROC Curves for each species (multi-class classification)
  - Confusion Matrix Heatmap
  - Feature Analysis via line plots for sepal/petal length and width
- Report: Classification metrics (Precision, Recall, F1-Score, Accuracy)
- Deployment: Served via Flask on port 5000

---

## Commands
docker run -p 5000:5000 --name iris iris:latest 
-> to build the image

docker run -p 5000:5000 --name iris iris:latest
-> to run the container

docker push ugnekniukstaite/2-task-big-data:tagname
-> to push the Docker image to a container registry

## Additional
When you run the Flask app, it will print two addresses to the console, but you should only use the first one: http://127.0.0.1:5000

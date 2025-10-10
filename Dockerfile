# start from Python base image
FROM python:3.11-slim

# create directory inside container
WORKDIR /app

# copy everything from project into container
COPY . /app

# install dependencies
RUN pip install -r requirements.txt

# expose the FastAPI port
EXPOSE 8000

# start the FastAPI app when container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

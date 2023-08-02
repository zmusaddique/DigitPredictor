FROM python:3.10.6

RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 5001

CMD ["python3", "main.py"]


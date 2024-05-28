FROM python:3.10.4-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN ls

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 
EXPOSE 5000

CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///app/mlruns & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
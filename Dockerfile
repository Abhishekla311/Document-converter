FROM python:3.10-slim

# पुरानी फ़ाइल रिप्लेसमेंट की जगह सीधा सोर्स लिस्ट को अपडेट करें
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# नेटवर्क टाइमआउट से बचने के लिए pip में --default-timeout फ्लैग जोड़ें
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt 

COPY . .

CMD ["uvicorn", "back:app", "--host", "0.0.0.0", "--port", "8000"]

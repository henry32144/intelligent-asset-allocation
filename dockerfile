FROM python:3.7
WORKDIR /Project/test

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "-c", "./gunicorn.conf.py"]
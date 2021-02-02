FROM continuumio/anaconda3:4.4.0
COPY . /App
EXPOSE 5000
WORKDIR ./App
RUN pip install --upgrade pip
RUN pip install --no-chche-dir -r requirements.txt
CMD python main.py
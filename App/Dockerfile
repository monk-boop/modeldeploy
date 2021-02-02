FROM continuumio/anaconda3:4.4.0
COPY ./App
EXPOSE 5000
WORKDIR ./App
RUN pip install -r requirements.txt
CMD python main.py
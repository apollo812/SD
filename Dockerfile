FROM python:3.10.2-slim-bullseye
# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
COPY . /txt2img
WORKDIR /txt2img
RUN pip install -r requirements.txt
EXPOSE 8000
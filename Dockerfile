FROM python:3.10-slim
WORKDIR /code
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
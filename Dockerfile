FROM python:3.10-slim
WORKDIR /code
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
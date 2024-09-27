uvicorn src.api.main:app --root-path ${BASE_PATH:-""} --port ${PORT:-8001} --host 0.0.0.0 --reload

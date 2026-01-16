FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app

COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./README.md .
COPY ./.python-version .
COPY ./pocket_tts ./pocket_tts

RUN uv run pocket-tts serve --help && \
    rm -rf /root/.cache/uv

EXPOSE 8000

CMD ["uv", "run", "pocket-tts", "serve", "--host", "0.0.0.0", "--port", "8000"]

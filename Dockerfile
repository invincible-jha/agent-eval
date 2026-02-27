FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/aumos-ai/agent-eval"
LABEL org.opencontainers.image.description="agent-eval: Multi-run statistical evaluation framework for AI agents"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="AumOS"

WORKDIR /app

# Install package
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir . && rm -rf /root/.cache

# Create non-root user
RUN useradd -m -s /bin/bash aumos
USER aumos

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["agent-eval"]
CMD ["--help"]

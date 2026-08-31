FROM node:20-bookworm-slim AS ui-build
WORKDIR /ui
COPY frontend/stage1b-monitor/package*.json ./
RUN npm ci
COPY frontend/stage1b-monitor/ ./
ARG VITE_TARCA_RUNTIME_LABEL="Stage 2 v1"
RUN npm run build

FROM pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/tarca/deploy/stage2/py310:/opt/tarca/src \
    TARCA_EXECUTION_KIND=stage2-v1 \
    TARCA_RUNTIME_STATIC_ROOT=/opt/tarca/frontend/stage1b-monitor/dist
RUN groupadd --gid 10001 tarca && useradd --uid 10001 --gid tarca --create-home tarca
WORKDIR /opt/tarca
COPY deploy/stage2/requirements-server.lock /opt/tarca/deploy/stage2/
COPY deploy/stage2/wheelhouse/ /opt/tarca/wheelhouse/
RUN python -m pip install --no-cache-dir --no-index --find-links=/opt/tarca/wheelhouse \
      --require-hashes -r /opt/tarca/deploy/stage2/requirements-server.lock \
    && python -c "import torch; assert torch.__version__.split('+')[0] == '2.2.2'"
COPY src/ /opt/tarca/src/
COPY configs/ /opt/tarca/configs/
COPY scripts/ /opt/tarca/scripts/
COPY deploy/stage2/ /opt/tarca/deploy/stage2/
COPY third_party/stage2/ /opt/tarca/third_party/stage2/
COPY --from=ui-build /ui/dist/ /opt/tarca/frontend/stage1b-monitor/dist/
RUN mkdir -p /opt/tarca/artifacts /opt/tarca/logs \
    && chown -R tarca:tarca /opt/tarca/artifacts /opt/tarca/logs \
    && chmod 0555 /opt/tarca/deploy/stage2/*.sh
USER tarca
ENTRYPOINT ["/opt/tarca/deploy/stage2/entrypoint.sh"]

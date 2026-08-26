FROM node:20-bookworm-slim AS ui-build
WORKDIR /ui
COPY frontend/stage1b-monitor/package*.json ./
RUN npm ci
COPY frontend/stage1b-monitor/ ./
RUN npm run build

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/tarca/deploy/stage1b/py310:/opt/tarca/src \
    TARCA_STAGE1B_SOURCE_CACHE_ROOT=/opt/tarca/official_sources \
    TARCA_STAGE1B_DATABASE=/opt/tarca/artifacts/stage1b/runtime/execution.sqlite3 \
    TARCA_STAGE1B_STATIC_ROOT=/opt/tarca/frontend/stage1b-monitor/dist
WORKDIR /opt/tarca
COPY deploy/stage1b/requirements-server.lock /tmp/requirements-server.lock
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --require-hashes --no-cache-dir -r /tmp/requirements-server.lock \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin tarca
COPY . /opt/tarca
COPY --from=ui-build /ui/dist /opt/tarca/frontend/stage1b-monitor/dist
RUN chmod 0555 /opt/tarca/deploy/stage1b/entrypoint.sh \
    && mkdir -p /opt/tarca/artifacts/stage1b/runtime /opt/tarca/official_sources \
    && chown -R tarca:tarca /opt/tarca/artifacts/stage1b /opt/tarca/official_sources
USER tarca
ENTRYPOINT ["/opt/tarca/deploy/stage1b/entrypoint.sh"]
CMD ["status", "--empty-ok"]

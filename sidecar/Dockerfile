FROM node:22-alpine

# better-sqlite3 needs a build toolchain at install time on alpine
RUN apk add --no-cache --virtual .build python3 make g++ \
 && mkdir -p /app /data /app/creds \
 && addgroup -S sidecar && adduser -S -G sidecar sidecar

WORKDIR /app
COPY --chown=sidecar:sidecar package.json package-lock.json* ./
RUN npm ci --omit=dev --no-audit --no-fund \
 && apk del .build

COPY --chown=sidecar:sidecar . .

USER sidecar

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DB_PATH=/data/sidecar.db
ENV CREDS_DIR=/data/creds

# /data should be a mounted volume so the SQLite mirror + Agent creds survive restarts.
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD wget -q -O- http://127.0.0.1:8000/healthz || exit 1

CMD ["node", "server.js"]

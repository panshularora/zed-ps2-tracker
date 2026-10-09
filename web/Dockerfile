# Frontend image: the built web app on nginx, /api proxied to the backend service.
# Build context is the repo root (the app imports viewer/src through the @viewer alias):
#   docker build -f web/Dockerfile -t zed-ps2-web .
FROM node:20-slim AS build
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY viewer/src/ /src/viewer/src/
COPY web/ ./
RUN npm run build

FROM nginx:1.27-alpine
ENV API_UPSTREAM=http://backend:8000
COPY web/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /src/web/dist/ /usr/share/nginx/html/
EXPOSE 80

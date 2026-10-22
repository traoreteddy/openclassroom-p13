# Étape 1 : construction de l'application Angular (production)
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# La bibliothèque ngx-chess-board (projets/) est compilée dans dist/ avant l'application,
# qui la résout par le chemin « dist/ngx-chess-board » (tsconfig.json)
RUN npm run build:board && npx ng build app --configuration production

# Étape 2 : serveur statique nginx, qui relaie /api vers le backend
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/app/browser /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1

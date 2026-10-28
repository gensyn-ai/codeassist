FROM node:22-alpine AS base

# Install dependencies
FROM base AS deps
WORKDIR /app

# Needed for native modules (e.g., utf-8-validate via ws) during npm ci
RUN apk add --no-cache python3 make g++

COPY package.json package-lock.json ./
RUN npm ci


# Build frontend server
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

ARG NEXT_PUBLIC_TESTER_URL
ENV NEXT_PUBLIC_TESTER_URL=$NEXT_PUBLIC_TESTER_URL

ARG NEXT_PUBLIC_STATE_SERVICE_URL
ENV NEXT_PUBLIC_STATE_SERVICE_URL=$NEXT_PUBLIC_STATE_SERVICE_URL

ARG NEXT_PUBLIC_POLICY_MODELS_URL
ENV NEXT_PUBLIC_POLICY_MODELS_URL=$NEXT_PUBLIC_POLICY_MODELS_URL

ARG NEXT_PUBLIC_ALCHEMY_API_KEY
ENV NEXT_PUBLIC_ALCHEMY_API_KEY=$NEXT_PUBLIC_ALCHEMY_API_KEY


ARG NEXT_PUBLIC_SIMULATION_MODE
ENV NEXT_PUBLIC_SIMULATION_MODE=$NEXT_PUBLIC_SIMULATION_MODE

ARG NEXT_PUBLIC_ZERO_STYLE_MODE
ENV NEXT_PUBLIC_ZERO_STYLE_MODE=$NEXT_PUBLIC_ZERO_STYLE_MODE

RUN npm run build


FROM base AS runner

RUN apk upgrade --no-cache ca-certificates ca-certificates-bundle \
    && apk version -t "$(apk info -e -v ca-certificates-bundle | sed 's/^ca-certificates-bundle-//')" 20261020-r0 | grep -qv '<'

WORKDIR /app

RUN apk --no-cache add curl

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000

ENV HOSTNAME="0.0.0.0"

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/healthz || exit 1

CMD ["node", "server.js"]

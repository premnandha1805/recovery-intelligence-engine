# ==============================================================================
# Stage 1: Build TypeScript Application
# ==============================================================================
FROM node:22-alpine AS build

WORKDIR /app

# Copy dependency specifications
COPY package*.json ./

# Install all dependencies (including devDependencies for TypeScript build)
RUN npm ci

# Copy configuration and source files
COPY nest-cli.json tsconfig*.json ./
COPY src/ ./src/

# Compile TypeScript to dist/
RUN npm run build

# Prune devDependencies to keep only production dependencies
RUN npm prune --omit=dev

# ==============================================================================
# Stage 2: Minimal Production Runtime
# ==============================================================================
FROM node:22-alpine AS production

ENV NODE_ENV=production \
    PORT=3000

WORKDIR /app

# Run as non-root built-in node user
USER node

# Copy production node_modules and built dist from build stage
COPY --chown=node:node --from=build /app/node_modules ./node_modules
COPY --chown=node:node --from=build /app/package.json ./package.json
COPY --chown=node:node --from=build /app/dist ./dist

# Expose NestJS API port
EXPOSE 3000

# Start NestJS production application
CMD ["node", "dist/main"]

FROM node:16.18.0

WORKDIR /src/evolution-web

COPY package*.json ./

RUN npm ci

COPY . .

RUN npm run build

CMD ["npm", "run", "server:start"]

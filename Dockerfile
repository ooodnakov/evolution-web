FROM node:16.18.0

WORKDIR /src
# RUN git clone https://github.com/ivan-work/evolution-web
COPY . /src/evolution-web
WORKDIR /src/evolution-web
RUN npm i
RUN cp .env.sample .env
RUN sed -i 's/PORT=/PORT=7080/' .env
RUN sed -i 's/NODE_ENV=/NODE_ENV=production/' .env
RUN sed -i 's/JWT_SECRET=/JWT_SECRET=de1c93044409424e6c95b4eaa726b493ee4d623974e210671ea12b77ca55edf0/' .env
RUN sed -i 's/ANALYTICS_DASHBOARD_PATH=.*/ANALYTICS_DASHBOARD_PATH=analytics/' .env
RUN sed -i 's/ANALYTICS_PG_HOST=.*/ANALYTICS_PG_HOST=analytics-db/' .env
RUN sed -i 's/ANALYTICS_PG_PORT=.*/ANALYTICS_PG_PORT=5432/' .env
RUN sed -i 's/ANALYTICS_PG_DATABASE=.*/ANALYTICS_PG_DATABASE=evolution_analytics/' .env
RUN sed -i 's/ANALYTICS_PG_USER=.*/ANALYTICS_PG_USER=evolution/' .env
RUN sed -i 's/ANALYTICS_PG_PASSWORD=.*/ANALYTICS_PG_PASSWORD=evolution_secret/' .env
ENV NODE_ENV=production
ENV ANALYTICS_DASHBOARD_PATH=analytics \
    ANALYTICS_PG_HOST=analytics-db \
    ANALYTICS_PG_PORT=5432 \
    ANALYTICS_PG_DATABASE=evolution_analytics \
    ANALYTICS_PG_USER=evolution \
    ANALYTICS_PG_PASSWORD=evolution_secret
RUN npm run build
CMD npm run server:start

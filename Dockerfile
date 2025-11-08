FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        openjdk-21-jre-headless \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && java -version

ENV JAVA_HOME="/usr/lib/jvm/java-21-openjdk-arm64"
ENV APP caravan_prediction_app

RUN mkdir -p ./${APP}
WORKDIR ./${APP}

COPY caravan_prediction_app/ ./caravan_prediction_app/
COPY src/ ./src/
COPY constraints.txt ./
COPY Makefile ./

RUN pip install --upgrade pip \
    && pip install setuptools --upgrade \
    && pip install --upgrade --use-feature=fast-deps -r constraints.txt

EXPOSE 8080

CMD make launch_api

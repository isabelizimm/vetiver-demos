# Generated by the vetiver package; edit with care

FROM rocker/r-ver:4.1.1
ENV RENV_CONFIG_REPOS_OVERRIDE https://packagemanager.rstudio.com/cran/latest

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
  git \
  libcurl4-openssl-dev \
  libgdal-dev \
  libgit2-dev \
  libicu-dev \
  libsodium-dev \
  libssl-dev \
  libxml2-dev \
  make \
  zlib1g-dev

COPY renv.lock renv.lock
RUN Rscript -e "install.packages('renv')"
RUN Rscript -e "renv::restore()"
COPY plumber.R /opt/ml/plumber.R

ENTRYPOINT ["R", "-e", "pr <- plumber::plumb('/opt/ml/plumber.R'); pr$run(host = '0.0.0.0', port = 8000)"]

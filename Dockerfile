FROM mambaorg/micromamba

COPY --chown=$(whoami):$(whoami) environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
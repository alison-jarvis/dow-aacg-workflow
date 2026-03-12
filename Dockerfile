FROM mambaorg/micromamba

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/entrypoint.sh"]

CMD ["/bin/bash"]
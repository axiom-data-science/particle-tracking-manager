FROM mambaorg/micromamba:2.0.5-debian12

ENV PROJECT_NAME=particle-tracking-manager
ENV PROJECT_ROOT=/opt/particle-tracking-manager

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes && \
    rm /tmp/environment.yml

COPY --chown=$MAMBA_USER:$MAMBA_USER particle_tracking_manager $PROJECT_ROOT/particle_tracking_manager
COPY --chown=$MAMBA_USER:$MAMBA_USER tests $PROJECT_ROOT/tests
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml setup.cfg conftest.py README.md $PROJECT_ROOT/

WORKDIR $PROJECT_ROOT/

ENTRYPOINT ["micromamba", "run", "python", "-m", "particle_tracking_manager.cli"]

# MettleBench worker/grader environment (one shared image for all admitted tasks).
# Upstream declares "Python 3.12+, git" plus django/pypdf/cryptography for a few
# evaluators; the remaining packages/tools are the ones the shipped workspaces,
# their tests and helper scripts import or invoke (static inventory, baseline
# evaluator runs and witness development). Worker and private grader run from
# this same image; the private verifier is copied only into a disposable
# grading container. Python packages are installed from the exact lock
# (python-lock.txt, resolved 2026-09-23 from requirements.txt). The default image
# omits the torch stack (python-lock-torch.txt); build with --build-arg WITH_TORCH=1
# for the task that needs it. Fewer files keep per-step R capture (docker diff) fast.
FROM python:3.12-slim-trixie
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN dpkg --add-architecture amd64 \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      git git-filter-repo openssl ca-certificates build-essential file sqlite3 \
      xz-utils bzip2 zip unzip patch diffutils procps less make \
      rsync bc jq zstd lz4 xxd bsdextrautils \
      libc6:amd64 libgcc-s1:amd64 zlib1g:amd64 \
 && rm -rf /var/lib/apt/lists/*
ARG WITH_TORCH=0
COPY python-lock-base.txt python-lock-torch.txt /opt/mettlebench/
RUN if [ "$WITH_TORCH" = 1 ]; then sort -f /opt/mettlebench/python-lock-base.txt /opt/mettlebench/python-lock-torch.txt; \
    else cat /opt/mettlebench/python-lock-base.txt; fi > /opt/mettlebench/python-lock.txt \
 && pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu -r /opt/mettlebench/python-lock.txt \
 && pip freeze | sort -f > /opt/mettlebench/pip-freeze.txt \
 && sort -f /opt/mettlebench/python-lock.txt | diff - /opt/mettlebench/pip-freeze.txt
# x86-64 floating-point parity: aarch64 GCC fuses multiply-adds (FMA) by default at -O2,
# which the authors' x86-64 toolchain does not; evaluators compare exact digests of
# compiled numeric kernels. Default -ffp-contract=off unless a build sets it explicitly.
RUN SPECS="$(dirname "$(gcc -print-libgcc-file-name)")/specs" \
 && gcc -dumpspecs | sed '/^\*cc1_options:$/{n;s/^/%{!ffp-contract=*:-ffp-contract=off} /}' > "$SPECS" \
 && printf 'double f(double a,double b,double c){return a*b+c;}\n' > /tmp/fma.c \
 && ! gcc -O2 -S -o - /tmp/fma.c | grep -q fmadd \
 && gcc -O2 -ffp-contract=fast -S -o - /tmp/fma.c | grep -q fmadd \
 && rm /tmp/fma.c
RUN git config --system user.name "MettleBench Worker" \
 && git config --system user.email "worker@mettlebench.invalid" \
 && git config --system init.defaultBranch main \
 && git config --system --add safe.directory '*'
WORKDIR /workspace

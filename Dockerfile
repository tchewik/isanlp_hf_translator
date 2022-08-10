FROM inemo/isanlp_base

RUN apt update
RUN apt install libffi-dev

RUN rm -r /root/.pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN pyenv install 3.9.0
RUN pyenv global 3.9.0

COPY requirements.txt .
RUN pip install -U pip \
    && pip install git+https://github.com/IINemo/isanlp.git \
    && pip install -r requirements.txt

COPY processor_hf_translator.py .
COPY pipeline_object.py .

CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline", "--no_multiprocessing", "True"]

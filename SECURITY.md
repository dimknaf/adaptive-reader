# Security

## Supported versions

Only the latest version on the `main` branch is supported.

## The demo server is a local tool

`serve.py` is meant for one person on one machine:
- **Local only:** it binds to `127.0.0.1` and accepts JSON requests only.
- **No authentication.**
- **One run at a time,** on the model server set by `CPM_URL`.
- **It saves every question and trace** under `results/web_<bundle>/`.

Do not expose it to a network. Put it behind your own authentication if you need remote access.

## Secrets

The project needs no API keys or credentials. The only settings are the model server's address (`CPM_URL`) and the
served model name (`CPM_MODEL`).

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the **Security** tab of this repository, then choose
**Report a vulnerability**. Please do not open a public issue for security problems.

"""Installing detection models from the maintainer's list (spec 030).

The package never imports NiceGUI or ``gui``: the install worker
(``python -m installer.worker``) runs it without the interface, so a failure
inside conda cannot take the app down with it.
"""

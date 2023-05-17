#!/bin/bash
python3 -m pip install -r requirements.txt && cd ./src && FLASK_APP=app.py FLASK_DEBUG=1 TEMPLATES_AUTO_RELOAD=1 python3 -m flask run
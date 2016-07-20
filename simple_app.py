"""TODO(grod): DO NOT SUBMIT without one-line documentation for simple_app.

TODO(grod): DO NOT SUBMIT without a detailed description of simple_app.
"""

import pdb
from flask import Flask, request
from transform import PICKLEFN, DATAPERSISTENCE, load_model, load_sparse_csr
import logging 

app = Flask(__name__)
myLogger = logging.getLogger()
@app.route("/", methods = ["POST"])
def handle_txt():
    myLogger.info("hit the / endpoint and in the handler")
    title = request.form.get('title', "")
    body = request.form.get('body', "")
    hasher = load_sparse_csr(DATAPERSISTENCE)[-1].item()
    mod = load_model(PICKLEFN)
    vec = hasher.transform([title+" "+body])
    pdb.set_trace()
    label = mod.predict(vec)[0]
    return label


app.run()

"""TODO(grod): DO NOT SUBMIT without one-line documentation for simple_app.

TODO(grod): DO NOT SUBMIT without a detailed description of simple_app.
"""

from flask import Flask, request
from transform import PICKLEFN, DATAPERSISTENCE, load_model, load_sparse_csr, tokenize_stem_stop

app = Flask(__name__)
@app.route("/", methods = ["GET", "POST"])
def handle_txt():
    if request.method == "POST":
      title = request.form.get('title', "")
      body = request.form.get('body', "")
    else:
      title = request.args.get('title', "")
      body = request.args.get('body', "")
    tokens = tokenize_stem_stop(" ".join([title, body]))
    hasher = load_sparse_csr(DATAPERSISTENCE)[-1].item()
    mod = load_model(PICKLEFN)
    vec = hasher.transform([tokens])
    label = mod.predict(vec)[0]
    return label
app.run(host="0.0.0.0")

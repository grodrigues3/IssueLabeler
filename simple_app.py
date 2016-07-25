from flask import Flask, request
from transform import DATAPERSISTENCE, load_model, load_sparse_csr, tokenize_stem_stop

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def handle_txt():
    team_mod = "pkld/trained_teams_model.pkl"
    comp_mod = "pkld/trained_components_model.pkl"
    if request.method == "POST":
      title = request.form.get('title', "")
      body = request.form.get('body', "")
    else:
      title = request.args.get('title', "")
      body = request.args.get('body', "")
    tokens = tokenize_stem_stop(" ".join([title, body]))
    hasher = load_sparse_csr(DATAPERSISTENCE)[-1].item()
    team_mod = load_model(team_mod)
    comp_mod = load_model(comp_mod)
    vec = hasher.transform([tokens])
    tlabel = team_mod.predict(vec)[0]
    clabel = comp_mod.predict(vec)[0]
    return ",".join([tlabel, clabel])

app.run(host="0.0.0.0")

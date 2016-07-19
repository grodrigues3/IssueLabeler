"""TODO(grod): DO NOT SUBMIT without one-line documentation for simple_app.

TODO(grod): DO NOT SUBMIT without a detailed description of simple_app.
"""

from flask import Flask, request
import logging
import pdb
app = Flask(__name__)
myLogger = logging.getLogger()


@app.route("/", methods = ["POST"])
def handle_txt():
    title = request.args.get('title', "")
    body = request.args.get('body', "")
    pdb.set_trace()
    toRet = "Hello World " + title + body 
    return toRet


app.run()

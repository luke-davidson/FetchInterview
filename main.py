# Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("template.html")






# Run
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
# Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Import and create model
from model import Modeler
model = Modeler()
model()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("month.html")

@app.route("/inputmonth", methods=["GET", "POST"])
def inputmonth():
    if request.method == "GET":
        return render_template("month.html")
    if request.method == "POST":
        response = request.form.get("month").split("-")
        # month_num, month_name = response.split("-")
        # pred = model.predict(int(month_num))
        return render_template("output.html", pred=response)




# Run
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
"""
Main Flask communication script.

author: Luke Davidson
        davidson.luked@gmail.com
        (978) 201-2693
"""

# Import dependencies
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Import and create model
from model import Modeler
model = Modeler()
model()

# Create Flask app
app = Flask(__name__)

# Create home page with month input
@app.route("/")
def index():
    return render_template("month.html")

# Render month output 
@app.route("/inputmonth", methods=["GET", "POST"])
def inputmonth():
    if request.method == "GET":
        return render_template("month.html")
    if request.method == "POST":
        # Obtain and format respose
        response = request.form.get("month").split("-")
        month_num, month_name = response

        # Input in to model
        pred = model.predict(int(month_num))

        # Render output
        return render_template("output.html", pred=pred, mon=month_name)

# Run
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
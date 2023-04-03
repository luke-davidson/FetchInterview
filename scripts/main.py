# # Import libraries
# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Web App Success Luke D!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int("8000"))
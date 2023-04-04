# Fetch - MLE Take Home Assignment
This repo holds all files created to create a future monthly receipt scanning prediction regression model.
author: Luke Davidson - davidson.luked@gmail.com - (978) 201-2693

# HOW TO RUN
This repo is set up to be built and run locally via a Docker container.
1. Clone repo locally and ensure docker is installed
2. Navigate to the cloned repo folder via the terminal
3. Run the command [`docker build -t <your_docker_username>/<your_docker_image_name>:latest .`] on the command line.
    * replace [`<your_docker_username>`] and [`<your_docker_image_name>`] with your custom entries to ensure a unique Docker image name is created.
    * It may take up to about a minute or so to build.
4. Run the command [`docker container run -d -p 8000:8000 <your_docker_username>/<your_docker_image_name>:latest`] from the terminal after the image has been built.
    * **Ensure [`<your_docker_username>`] and [`<your_docker_image_name>`] are the same as step 3!**
5. Navigate to the url [`localhost:8000`] in your web browser

# File descriptions
[`main.py`](https://github.com/luke-davidson/FetchInterview/blob/main/main.py)
- Main communication script with the web app. Receives input from the web app via the [`month.html`] file, passes in the input to the model, and passes the output back to the web app via the [`output.html`] file.

[`model.py`](https://github.com/luke-davidson/FetchInterview/blob/main/model.py)
- Main model script. The script creates an optimized Ordinary Least Squares (OLS) Regression model via k-fold cross validation on the 2021 receipt scan data. The theoretical equation derived from minimizing the Sum of Squared Error (SSE) is used to calculate regression parameter unknowns. These unknowns are tested against the validation data and the parameters that yield the lowest validation error are selected as the final model parameters and are used to calculate future predictions. Implementations of cross validation, OLS Regression, prediction and model selection are naive to show understanding of ML processes.

[`templates/template.html`](https://github.com/luke-davidson/FetchInterview/blob/main/templates/template.html)
- Main .html template to create web app heading.

[`templates/month.html`](https://github.com/luke-davidson/FetchInterview/blob/main/templates/month.html)
- .html template used to receive drop down input of what month of 2022 the user would like to obtain a receipt scan prediction of.

[`templates/output.html`](https://github.com/luke-davidson/FetchInterview/blob/main/templates/output.html)
- .html template used to render and model prediction output, visual and return button.

[`Dockerfile`](https://github.com/luke-davidson/FetchInterview/blob/main/Dockerfile)
- Dockerfile used to ensure the docker image, environment and container are built correctly.

[`requirements.txt`](https://github.com/luke-davidson/FetchInterview/blob/main/requirements.txt)
- File including the necessary python dependencies needed to correctly run [`main.py`] and [`model.py`].

[`data_daily.csv`](https://github.com/luke-davidson/FetchInterview/blob/main/data_daily.csv)
- Raw 2021 daily receipt scan data in .csv format.
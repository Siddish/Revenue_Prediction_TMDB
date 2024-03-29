# Revenue_Prediction_TMDB
This project is chosen from Kaggle. The dataset and project details are as mentioned here: (https://www.kaggle.com/c/tmdb-box-office-prediction).

The mode of deployment for this project are as follows:
1. Create and launch an AWS instance through EC2 which is your deployment server
2. Install Python on this deployment server
3. Install git bash through command prompt on the deployment server
4. Then clone the entire repo onto the deplopyment server
5. On the command prompt of your deployment server, install the packages as mentioned in the requirements.txt file in the repo using command pip install -r <path-to-requirements.txt-file>
6. Run the script production_model.py first which creates the model and generates the two pickle files model.pkl and model_columns.pkl using command python <script-name.py>
7. Run the api.py script which deploys the model as an API
8. Use the IP address generated by the script in the output to connect to the deployment server through Postman
9. On the Postman app on your local computer use IP address and add the word predict at the end to interact with the server:
For example: if IP address generated is 127.0.0.1/12345
On Postman use 127.0.0.1/12345/predict as the POST request
10. Copy the sample input as provided in the api.py script in JSON format and paste it in the Body field of Postman UI
11. Change the format next to body as JSON if needed. 
12. Click on "Send" on Postman and you will see the output prediction as JSON at the response field on Postman

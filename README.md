Be sure to use the same version of the code as the version of the docs you're reading. You probably want the latest tagged version, but the default Git version is the master branch.

step-1:
Clone the repository

step-2:
Open shell or any command line and move to your project directory

step-3:
And install all required packages on your virtual enviroment
by "$ pip install -e ."

step-4:
We use Fast API here so we use "uvicorn main:app --reload" command to run our server here main is your file where you code your function and app is your reference of your FastAPI class

step-5:
You could run test_request.py for testing your project or you could use
third party application i.e."Postman" and make a post request. 

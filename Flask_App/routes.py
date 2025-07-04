from Flask_App import app

@app.route("/")
def home():
    return  "Welcome to the fixed Flask Structure"
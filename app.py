from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import File, UploadFile
from src.components.predict import Prediction
from fastapi import FastAPI, Request
import uvicorn

# Create a new instance of the FastAPI app
app = FastAPI()

# Mount the static directory for serving files
app.mount("/static", StaticFiles(directory="static"))

# Initialize Jinja2Templates for rendering HTML templates
TEMPLATES = Jinja2Templates(directory='templates')

# Initialize an empty list to store searched images
searchedImages = []

# Create a new instance of the Prediction class
predict_pipe = Prediction()

# Define the index route
@app.get("/", status_code=200)
@app.post("/")
async def index(request: Request):
    """
    Description : This Route loads the index.html
    """
    # Render the index.html template with the request object as context
    return TEMPLATES.TemplateResponse(name='index.html', context={"request": request})

# Define the image upload route
@app.post('/image')
async def upload_file(file: UploadFile = File(...)):
    """
    Description : This Route loads the predictions in a list which will be listed on webpage.
    """
    global searchedImages, predict_pipe
    try:
        # Check if the prediction pipeline is loaded
        if predict_pipe:
            # Read the contents of the uploaded file
            contents = file.file.read()

            # Run the predictions using the loaded pipeline
            searchedImages = predict_pipe.run_predictions(contents)

            # Return a success message
            return {"message": "Prediction Completed"}
        else:
            # If the pipeline is not loaded, return an error message
            return {"message": "First Load Model in Production using reload_prod_model route"}
    except Exception as e:
        # Catch any exceptions and return an error message
        return {"message": f"There was an error uploading the file {e}"}

# Define the reload route
@app.post('/reload')
def reload():
    """
    Description : This Route resets the predictions in a list for reload.
    """
    global searchedImages

    # Reset the searched images list
    searchedImages = []

    # Return a success message
    return

# Define the reload prod model route
@app.get('/reload_prod_model')
def reload():
    """
    Description : This Route is Event Triggered or owner controlled to update
                    the model in prod with minimal downtime.
    """
    global predict_pipe
    try:
        # Delete the existing prediction pipeline
        del predict_pipe

        # Create a new instance of the Prediction class
        predict_pipe = Prediction()

        # Return a success message
        return {"Response": "Successfully Reloaded"}
    except Exception as e:
        # Catch any exceptions and return an error message
        return {"Response": e}

# Define the gallery route
@app.get('/gallery')
async def gallery(request: Request):
    """
    Description : This Route lists all the predicted images on the gallery.html listing depends on prediction.
    """
    global searchedImages

    # Render the gallery.html template with the request object and searched images list as context
    return TEMPLATES.TemplateResponse('gallery.html', context={"request": request, "length": len(searchedImages),
                                                            "searchedImages": searchedImages})

# Run the app if this script is executed directly
if __name__ == "__main__":
    # Run the app on host 0.0.0.0 and port 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)

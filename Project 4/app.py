from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from fpdf import FPDF

# Load all the environment variables
load_dotenv()

# Configure the Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load your trained CNN model
cnn_model = tf.keras.models.load_model('trained_model.h5')

# Define your class names corresponding to the output labels of the CNN
class_names = [
    "Apple", "Banana", "Carrot", "Grapes", "Mango", "Onion", "Orange", "Peas",
    "Potato", "Tomato", "Strawberry", "Watermelon", "Cabbage", "Broccoli", 
    "Eggplant", "Lettuce", "Spinach", "Zucchini", "Pineapple", "Blueberry", 
    "Kiwi", "Papaya", "Beetroot", "Bell Pepper", "Cauliflower", "Garlic", 
    "Ginger", "Radish", "Squash", "Cucumber", "Celery", "Chili", 
    "Corn", "Cherries", "Peach", "Plum", "Pomegranate", "Almond", 
    "Walnut"
]

# Function to preprocess the uploaded image for CNN
def preprocess_image(image):
    # Convert the image to RGB (to ensure it has 3 channels)
    image = image.convert("RGB")
    image = np.array(image)
    
    # Resize to match CNN input
    image = cv2.resize(image, (64, 64))
    
    # Expand dimensions to add batch size (1, 64, 64, 3)
    img_array = np.expand_dims(image, axis=0)
    
    return img_array / 255.0  # Normalize to [0, 1]

# Function to get food class prediction
def predict_class(image):
    img_array = preprocess_image(image)
    predictions = cnn_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index
    predicted_class_label = class_names[predicted_class_index]  # Get the class name
    return predicted_class_label  # Return the predicted class label

# Function to load Google Gemini Pro Vision API and get response
def get_gemini_response(input_prompt, image, user_data):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_prompt, image[0], user_data])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to generate PDF report
def generate_pdf_report(gemini_response, filename="diet_workout_report.pdf"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Diet and Workout Report", ln=True, align="C")

    # Split Gemini response into lines and write each line to the PDF
    for line in gemini_response.split('\n'):
        pdf.multi_cell(0, 10, line)

    # Save PDF to a file
    pdf.output(filename)

# Initialize our Streamlit app
st.set_page_config(page_title="Diet and Workout Recommendation")

st.header("Diet and Workout Recommendation")

# Taking user inputs
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Button for food recognition
if st.button("Recognize Food"):
    if image is not None:
        predicted_class_label = predict_class(image)
        st.write(f"Predicted Food: {predicted_class_label}")
    else:
        st.error("Please upload an image first.")

# Taking user inputs for recommendations
# Ingredients input for more accurate calorie estimation
ingredients = st.text_area("Enter ingredients and their measurements (e.g., 'strawberry: 30 grams, blueberry: 40 grams, kivi: 20 grams, mango: 40 grams, orange: 50 grams')")
age = st.number_input("Enter your age:", min_value=1, value=20)
gender = st.selectbox("Select your gender:", ["Male", "Female"])
weight = st.number_input("Enter your weight (kg):", min_value=10, value=55)
height = st.number_input("Enter your height (cm):", min_value=100, value=164)
waist = st.number_input("Enter your waist size (cm):", min_value=15, value=80)
neck = st.number_input("Enter your neck size(cm):", min_value=1, value=40)
activity = st.selectbox("How often do you workout weekly:", ["Sedentary: little or no exercise", "Light: exercise 1-3 times/week", "Moderate: exercise 4-5 times/week", "Active: daily exercise or intense exercise 3-4 times/week", "Very Active: intense exercise 6-7 times/week"])
diet_preference = st.selectbox("Diet Preference:", ["Veg", "Non-Veg"])
disease = st.selectbox("Select medical conditions (if none select 'none'):", ["None", "Diabetes", "Heart Failure", "Obesity", "Kidney Disease", "Cancer", "Strokes", "Alzheimer"])
region = st.text_input("Enter your region:", value="India")
allergies = st.text_input("Enter any allergies (if none, type 'None')", value="None")
food_type = st.selectbox("Meal Type:", ["Breakfast", "Lunch", "Dinner", "Snack"])

# Define the static prompt for the API
static_prompt = """
You are an expert in nutrition. You need to recognize the food and analyze the food items from the image,
calculate the estimated calorie  details of every food item calculate the estimated total calories intake in the following format:

table format rows and column 
    column Item, total calories, protein, carbs, fats, Fiber, Vitamins
    row Item 1, Item 2, Item 3,....

---- 
---- 

after that consider my input values provided age, gender, weight, height, diet_preference, disease, region, food_type and tell whether the food is healthy for me or not.
And then Provide me recommendations for Diet Recommendation (daily 4 meals measured calorie).
And calculate BMI and provide Workout Recommendations and workout plan for week based on my input values.
And provide additional tips
    > Food Recognition
        Predicted Food: "provide only name of the food"
    > Calorie Estimation 
    > food analysis
    > Diet Recommendation
        Meals recommendation (4 meals with measured calorie)
        1) Food to take
        2) Food to avoid
    > Fitness Report:
        1) BMI (Body Mass Index)
        2) Calculate BFP (Body fat percentage) using BFP = 86.010×log10(abdomen-neck) - 70.041×log10(height) + 36.76 
        3) Calculate Daily Calorie Intake
    > Workout Recommendation for week (table format: Day, Exercise, Duration)
    > Workout Exercises
    > Additional tips for fitness
"""

# Button for getting recommendations
if st.button("Get Recommendations!"):
    try:
        # Prepare user data as a string
        image_data = input_image_setup(uploaded_file)
        user_data = f"Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} cm, Waist: {waist} cm, Neck: {neck} cm, Activity: {activity}, Diet Preference: {diet_preference}, Disease: {disease}, Region: {region}, Allergies: {allergies}, Food Type: {food_type}, Ingredients: {ingredients}"
   
        # Get the response from Gemini API
        response = get_gemini_response(static_prompt, image_data, user_data)
        st.subheader("Personalized Diet and Workout Recommendation")
        st.write(response)

        # Generate PDF report
        pdf_filename = "diet_workout_report.pdf"
        generate_pdf_report(response, filename=pdf_filename)
        
        # Provide download link for the PDF
        with open(pdf_filename, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            st.download_button(label="Download Your Report", data=pdf_bytes, file_name=pdf_filename, mime='application/octet-stream')
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

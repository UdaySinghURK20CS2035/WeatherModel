# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

import requests
import joblib
from sklearn.preprocessing import StandardScaler

# Replace with your actual API key
api_key = st.secrets["API_KEY"]


def get_weather_data(city_name, api_key):
    """
    Fetch weather data for a specified city using the OpenWeatherMap API.

    Args:
        city_name (str): The name of the city to fetch the weather for.
        api_key (str): Your OpenWeatherMap API key.

    Returns:
        dict: A dictionary containing weather information or an error message.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"

    try:
        response = requests.get(complete_url)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e)}

    data = response.json()

    if response.status_code == 200:
        # Extracting required information
        temperature = data['main']['temp']  # Temperature in Celsius
        humidity = data['main']['humidity'] * 100  # Humidity percentage
        wind_speed_m_s = data['wind']['speed']  # Wind speed in m/s
        pressure = data['main']['pressure']  # Pressure in hPa (millibars)

        # Convert wind speed from m/s to km/h
        wind_speed_kmh = round(wind_speed_m_s * 3.6, 2)

        weather_info = {
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "Wind Speed (km/h)": wind_speed_kmh,
            "Pressure (millibars)": pressure
        }
        return weather_info
    else:
        return {"error": data.get("message", "Unknown error occurred")}


def model_prediction(weather_data):
    model_filename = "joblib_model/random_forest_model.joblib"
    # # Load the model from the file
    loaded_model = joblib.load(model_filename)
    print("Model loaded successfully")

    scaler = StandardScaler()
    label_encoder = joblib.load("joblib_model/label_encoder.joblib")
    # # Example of making a prediction with the loaded model
    new_data = [weather_data]
    print(new_data)

    # Example of making a prediction with the loaded model
    scaler.fit(new_data)
    new_data_scaled = scaler.transform(new_data)
    predicted_class = loaded_model.predict(new_data_scaled)
    print("\nPredicted class for new data with loaded model:", predicted_class)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]


# Function to convert PIL image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def main():
    # Load the image
    image = Image.open('media/WeatherWiz.png')
    image_base64 = image_to_base64(image)

    # Write the CSS for centering the image and tagline and adjusting the top margin
    center_css = """
    <style>
        .centered-content {
            display: block;
            text-align: center;
            width: 50%; /* Adjust the width as needed */
            margin-left: auto;
            margin-right: auto;
        }
        .centered-text {
            text-align: center;
            font-size: 20px; /* Adjust the font size as needed */
        }
        .content-container {
            margin-top: 20px; /* Adjust the top margin of the content container */
        }
    </style>
    """

    # Apply the CSS using st.markdown
    st.markdown(center_css, unsafe_allow_html=True)

    # Add a container for the content to adjust the top margin
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    # Display the image with the centered class
    st.markdown(
        f'<div class="centered-content"><img src="data:image/png;base64,{image_base64}" class="centered-image"></div>',
        unsafe_allow_html=True)

    # Display the centered caption
    st.markdown('<div class="centered-text">"From Clouds to Sunshine, We\'ve Got You Covered"</div>',
                unsafe_allow_html=True)

    # Close the content container
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar + Main panel
    st.sidebar.header('Input Options')

    # Add an input field in the sidebar
    user_input = st.sidebar.text_input("Enter City:", placeholder="For example London...")

    temperature = ""
    humidity = ""
    windspeed = ""
    pressure = ""
    weather_pred = ""

    # Add a button to submit the value
    if st.sidebar.button("Submit"):
        st.sidebar.write("Report Generated Successfully")
        weather = list(get_weather_data(user_input, api_key).values())
        temperature = weather[0]
        humidity = weather[1] / 100
        windspeed = weather[2]
        pressure = weather[3]
        weather_pred = model_prediction(weather)

    st.header("Weather Information")

    # Display the values associated with the fields
    st.write("Temperature (🌡):", temperature, "°C")
    st.write("Humidity:(🌥️)", humidity, "%")
    st.write("Windspeed:(💨)", windspeed, "km/hr")
    st.write("Pressure:(🕧)", pressure, "millibars")
    st.write("Current Weather expected to be:", weather_pred)


if __name__ == "__main__":
    main()

import streamlit as st
from streamlit_option_menu import option_menu
import time
import pickle
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import google.generativeai as genai

st.set_page_config(layout="wide")

# Navigation Bar
selected = option_menu(
    menu_title = None,
    options = ["Home", "Travel Recommendation", "Itinerary Planner"],
    icons = ["house-fill", "airplane-fill", "card-checklist"],
    menu_icon = "cast",
    default_index = 0,
    orientation = "horizontal",
)


###########################################################################################
# To display the home page
if selected == "Home":

    # st.image('Data/Pictures/Banner.jpg', use_container_width=True)

    # Main title
    st.title("🌍 Travel Recommendation System")

    # Description section
    st.markdown(
        '''
        <div style="text-align: justify;">
            Welcome to our Personalized Travel Recommendation System, your ultimate companion for creating unforgettable travel experiences. 
            Powered by advanced machine learning algorithms, our system analyzes your preferences, budget, and interests to provide tailored destination recommendations and optimized itineraries.
        </div>
        ''',
        unsafe_allow_html = True
    )
    # Add space
    st.markdown("")
    st.markdown(
        '''
        <div style="text-align: justify;">
            Whether you're seeking cultural adventures, relaxation, or thrilling activities, our platform ensures every recommendation aligns perfectly with your unique travel style. 
            With seamless integration of real-time data and user feedback, we bring you the most relevant and insightful suggestions to simplify your travel planning process.
        </div>
        ''',
        unsafe_allow_html = True
    ) 
    st.markdown("")

    # Key Features in Columns
    st.header("Why Choose Us?")
    st.markdown("")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('Data/Pictures/Tailored Recommendations.jpg', caption="Tailored Recommendations", use_container_width=True)
        st.markdown('''<div style="text-align: center; font-weight: bold;">Personalized Itineraries</div>''', unsafe_allow_html=True)

    with col2:
        st.image('Data/Pictures/Real-Time Updates.jpg', caption="Real-Time Updates", use_container_width=True)
        st.markdown('''<div style="text-align: center; font-weight: bold;">Up-to-date Travel Insights</div>''', unsafe_allow_html=True)

    with col3:
        st.image('Data/Pictures/Effortless Planning.jpg', caption="Effortless Planning", use_container_width=True)
        st.markdown('''<div style="text-align: center; font-weight: bold;">Simplified Travel Planning</div>''', unsafe_allow_html=True)

    # Highlighted Call-to-Action
    st.markdown("""
        ---
        **Plan smarter, Travel better, and Explore the world with confidence.**
        """)


###########################################################################################
# To display the travel page
if selected == "Travel Recommendation":

    st.title("Travel Recommendation System")
    # Import artifacts
    travel = pickle.load(open('artifacts/place_list.pkl', 'rb'))
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    city_name = pickle.load(open('artifacts/city_name.pkl', 'rb'))
    city_pivot = pickle.load(open('artifacts/city_pivot.pkl', 'rb'))
    final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))

    # Load the Excel file containing city links
    links_data = pd.read_excel("Data/Links.xlsx")

    # Define the tooltip CSS
    tooltip_css = """
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position the tooltip above the text */
        left: 50%;
        margin-left: -60px; /* Center the tooltip */
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """

    # Add the CSS to the Streamlit app
    st.markdown(tooltip_css, unsafe_allow_html=True)


    # Content Based
    def recommend(place):
        # Check if the place exists in the travel DataFrame
        if place not in travel['city'].values:
            return []  # Return an empty list or handle it as needed

        index = travel[travel['city'] == place].index[0]
        distances = sorted(list(enumerate(similarity[index])),reverse=True, key = lambda x:x[1])
        recommended_place_name = []
        #recommended_place_poster = []

        for i in distances[1:7]:
            c_id = travel.iloc[i[0]].c_id
            #recommended_place_poster.append(fetch_poster(c_id))
            recommended_place_name.append(travel.iloc[i[0]].city)

        return recommended_place_name

    # Collaborative Filtering
    def recommend_city(city_name):
        cities_list = []
        # Check if the city_name exists in the city_pivot index
        if city_name not in city_pivot.index:
            return []  # Return an empty list if the city is not found

        city_id = np.where(city_pivot.index == city_name)[0][0]
        distance, suggestion = model.kneighbors(city_pivot.iloc[city_id,:].values.reshape(1,-1), n_neighbors = 6)
    
        for i in range(len(suggestion)):
            cities = city_pivot.index[suggestion[i]]
            for j in cities:
                cities_list.append(j)

        return cities_list

    # Function to validate city names
    def is_valid_city(city):
        # Example criteria: names should be longer than one character and contain only letters
        return len(city) > 1 

    # Combine and filter city names
    place_list = set(filter(is_valid_city, city_name))  # Filtered unique cities from city_name
    place_list.update(filter(is_valid_city, travel['city'].values))  # Filtered unique cities from travel['city']
    place_list = list(place_list)  # Convert back to a list

    # Sort the place_list alphabetically
    place_list = sorted(place_list)

    # Get the total number of unique cities
    total_cities = len(place_list)

    # Display the total number of cities (if using Streamlit for example)
    st.write(f"Total number of cities: {total_cities}")

    selected_city = st.selectbox(
        "Type or select a City",
        place_list
    )

    # Function to get URL for a city
    def get_city_url(city_name):
        match = links_data[links_data['City'] == city_name]
        if not match.empty:
            return match.iloc[0]['URL']
        return "#"  # Return a default value if no match is found

    # Function to get information about a city
    def get_city_info(city_name):
        match = links_data[links_data['City'] == city_name]
        if not match.empty:
            return {
                'Country': match.iloc[0]['Country'],
                'Population': match.iloc[0]['Population'],
                'Area (sq mi)': match.iloc[0]['Area (sq mi)'],
            }
        return None  # Return a default value if no match is found

    if st.button('Show Recommendation'):
        st.write("Your Recommendations are: ")
        recommendation_cities = recommend_city(selected_city)
        recommended_place_name = recommend(selected_city)
        # Combine the two lists and remove duplicates
        combined_recommendations = set(recommended_place_name + recommendation_cities)

        # Remove the selected city if it's in the combined recommendations
        combined_recommendations.discard(selected_city)

        # Convert back to a list
        combined_recommendations = list(combined_recommendations)

        # Show total count of cities
        total_count = len(combined_recommendations)
        st.write(f"Count of recommended cities: {total_count}")

        # Check if there are any recommendations
        if not combined_recommendations:
            st.write("No recommendations available.")
        else:
            # Display the combined recommendations in rows of 5
            num_columns = 5
            cols = st.columns(num_columns)  # Create 5 columns

            for i, city in enumerate(combined_recommendations):
                col_index = i % num_columns  # Determine the column index
                with cols[col_index]:
                    url = get_city_url(city)  # Assuming you have a function to get the URL
                    city_info = get_city_info(city)  # Get city information
                    if city_info:  # Check if city_info is not None
                        tooltip_text = (
                            f"Country: {city_info.get('Country', 'N/A')}<br>"
                            f"Population: {city_info.get('Population', 'N/A')}<br>"
                            f"Area: {city_info.get('Area (sq mi)', 'N/A')} sq mi"
                        )
                    else:
                        tooltip_text = "No info available"
                
                    link_html = f'''
                    <div class="tooltip">
                        <a href="{url}" style="font-weight: bold; text-decoration: none; color: black;">{city}</a>
                        <span class="tooltiptext">{tooltip_text}</span>
                    </div>
                    '''
                    st.markdown(f'<div style="text-align: center; margin: 10px;">{link_html}</div>', unsafe_allow_html=True)

                # If we reach the last column, create a new row
                if col_index == num_columns - 1 and i < len(combined_recommendations) - 1:
                    cols = st.columns(num_columns)  # Create new columns for the next row
    

###########################################################################################
# To display the itinerary page
if selected == "Itinerary Planner":

    # Load the API Key
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Function to load Google Gemini Pro Model and get response
    def get_response(prompt, input):
        model = genai.GenerativeModel('gemini-1.5-flash-001')
        response = model.generate_content([prompt, input], stream=False)
        return response.text

    # Initialize the streamlit app
    # st.set_page_config(page_title="Planner: Discover and Plan your Culinary Adventures!")
    # st.image('Data/Pictures/logo.jpg', width=70)

    st.title("Itinerary Planner")
    st.subheader("Discover and Plan your Adventures!")

    # Create a select box for section choices
    section_choice = st.selectbox("Choose Section:", ("", "Trip Planner", "Accommodation", "Transport", "Food Preferences"))
    st.markdown("")

    # If the choice is empty
    if section_choice == "":
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image('Data/Pictures/Itiernary - 4.jpg', use_container_width=True)
            st.markdown('''<div style="text-align: center; font-weight: bold;">Trip Planner</div>''', unsafe_allow_html=True)

        with col2:
            st.image('Data/Pictures/Itiernary - 3.jpg', use_container_width=True)
            st.markdown('''<div style="text-align: center; font-weight: bold;">Accommodation</div>''', unsafe_allow_html=True)

        with col3:
            st.image('Data/Pictures/Itiernary - 2.jpg', use_container_width=True)
            st.markdown('''<div style="text-align: center; font-weight: bold;">Transport</div>''', unsafe_allow_html=True)

        with col4:
            st.image('Data/Pictures/Itiernary - 1.jpg', use_container_width=True)
            st.markdown('''<div style="text-align: center; font-weight: bold;">Food Preferences</div>''', unsafe_allow_html=True)

    # If the choice is Trip Planner
    if section_choice == "Trip Planner":
        # Prompt Template
        input_prompt_planner = """
        You are an expert Tour Planner and Travel Consultant. Your job is to create a highly personalized travel plan based on the following attributes:
        1. Location: The main destination(s) or cities provided by the user.
        2. Budget: Plan activities, accommodations, and meals within the specified budget range.
        3. Travel Dates (Duration): Include plans for the given number of days or estimate an appropriate duration if not provided. Suggest suitable travel dates or seasons.
        4. Travel Party: Cater to the group type (e.g., solo traveler, couple, family, or friends) and their specific needs or dynamics.
        5. Activities and Interests: Focus on the user's preferences such as adventure (e.g., hiking, water sports), relaxation (e.g., spa, beaches), cultural (e.g., museums, heritage sites), or nightlife and shopping.
        For the plan:
        - Suggest an optimal itinerary with day-wise recommendations for activities and places to visit.
        - Highlight hidden secrets, must-visit landmarks, and off-the-beaten-path gems.
        - Mention the best time or season to visit the destination.
        - Provide safety tips, sustainability tips, and any other special considerations.
        Return the response in markdown format for easy readability, with clear headings, subheadings, and a day-wise itinerary breakdown.
        """

        # Input Fields
        location = st.text_input("Enter the location:")
        budget = st.text_input("Enter your budget (e.g., Low, Medium, High):")
        travel_dates = st.text_input("Enter travel dates (e.g., 5 days, 1 week):")
        travel_party = st.selectbox("Select your travel party:", ["Solo", "Couple", "Family", "Friends"])
        activities_interests = st.multiselect("Select activities and interests:", ["Adventure", "Relaxation", "Cultural", "Shopping", "Nightlife"])

        # Button
        submit1 = st.button("Plan my Trip!")
        if submit1:
            # Construct the input plan based on user inputs
            input_plan = f"""
            Location: {location}
            Budget: {budget}
            Travel Dates: {travel_dates}
            Travel Party: {travel_party}
            Activities and Interests: {', '.join(activities_interests)}
            """
            
            response = get_response(input_prompt_planner, input_plan)
            st.subheader("Itinerary Planner: ")
            # Justify the response text using HTML and CSS
            justified_response = f'<div style="text-align: justify;">{response}</div>'
            st.markdown(justified_response, unsafe_allow_html=True)

    
    # If the choice is Accommodation
    if section_choice == "Accommodation":

        # Accommodation Preferences Prompt Template
        input_prompt_accommodation = """
        You are an expert Accommodation Advisor. Your primary goal is to provide tailored accommodation recommendations based on the user's input preferences.
        Consider the following factors:
        1. Location: The main destination(s) or cities provided by the user.
        2. Budget: Recommendations should align with the budget range provided (e.g., low, medium, high).
        3. Type of Accommodation: The user has specified their preferred type, such as Hotels, Hostels, Vacation Rentals, or Camping.
        4. Proximity to Attractions: Consider the user's proximity preferences, whether they prefer accommodations close to major attractions, in well-connected areas, or in quiet neighborhoods.
        5. Travel Party: Adjust recommendations to suit the travel party, such as solo travelers, couples, families, or groups of friends. For example:
            - Families: Family-friendly accommodations with extra amenities for kids.
            - Couples: Romantic settings or privacy-focused options.
            - Solo Travelers: Budget-friendly or dorm-style rooms.
            - Friends: Spacious and social settings like hostels or group vacation rentals.
        For the Accommodation:
        - Provide rating of the Hotels/Hostels/Vacation Rentals, or Camping (in a table format)
        - Top 5 hotels within the budget and proximity to Attractions given by the user with address and average cost per night (in a table format)
        Format the response in markdown for clear presentation.
        """

        # Input Fields
        location = st.text_input("Enter the location:")
        budget = st.text_input("Enter your budget (e.g., Low, Medium, High):")
        accommodation_type = st.selectbox("Select the type of accommodation:", ["Hotels", "Hostels", "Vacation Rentals", "Camping"])
        proximity_to_attractions = st.multiselect("Select proximity preferences:", ["Close to major attractions", "Well-connected areas", "Quiet neighborhoods"])
        travel_party = st.selectbox("Select your travel party:", ["Solo", "Couple", "Family", "Friends"])

        # Button
        submit2 = st.button("Find Accommodation!")
        if submit2:
            # Construct the input plan based on user inputs
            input_accommodation = f"""
            Location: {location}
            Budget: {budget}
            Accommodation Type: {accommodation_type}
            Proximity to Attractions: {', '.join(proximity_to_attractions)}
            Travel Party: {travel_party}
            """
        
            response = get_response(input_prompt_accommodation, input_accommodation)
            st.subheader("Accommodation Recommendations: ")
            # Justify the response text using HTML and CSS
            justified_response = f'<div style="text-align: justify;">{response}</div>'
            st.markdown(justified_response, unsafe_allow_html=True)


    # If the choice is Transport
    if section_choice == "Transport":

        # Transport Preferences Prompt Template
        input_prompt_transport = """
        You are a highly skilled Transport Advisor, dedicated to delivering personalized and efficient transport recommendations tailored to the user's specific preferences. When crafting your suggestions, consider the following key factors:
        1. Location:
            Take into account the primary destinations or cities specified by the user.
            Assess the geographical layout, accessibility, and connectivity between the locations.
        2. Mode of Transport:
            Offer a range of options, including car, train, flight, and bike, based on the user's priorities such as speed, convenience, or scenic experiences.
            Highlight any unique travel opportunities available at the specified location, such as scenic train routes or self-drive tours.
        3. Availability of Rental Services:
            Provide insights into rental options available, including car rentals, bike rentals, e-scooter rentals, and specialized options like RV or campervan rentals.
            Include details about the accessibility of rental services (e.g., availability near airports, train stations, or city centers) and their suitability for the user's itinerary.
        4. Public Transport Preferences:
            Offer recommendations for urban and intercity public transportation, such as metro systems, buses, or high-speed trains.
            Suggest eco-conscious options like electric buses or carpooling services for environmentally aware travelers.
            Highlight local experiences that can be gained from public transportation, such as trams in historic districts or scenic ferry rides.
        5. Tailoring the Suggestions:
            Ensure that your recommendations align with the travel party’s needs, whether they are solo travelers, couples, families, or groups of friends. For example:
            Family trips: Focus on safe and convenient modes of transport, such as car rentals or direct train connections.
            Eco-conscious travelers: Recommend options like electric car rentals or public transport networks with green certifications.
        Deliver your recommendations in a clear, user-friendly manner using markdown, ensuring key points are emphasized for quick comprehension. Additionally, include practical tips, such as links to transport websites, ticket booking platforms, or rental services. Tailor your tone to be professional, approachable, and engaging, ensuring users feel guided and confident in their travel planning.
        """

        # Input Fields
        location = st.text_input("Enter the location for transport recommendations:")
        mode_of_transport = st.selectbox("Select the mode of transport:", ["Car", "Train", "Flight", "Bike"])
        rental_services = st.multiselect("Select rental services available:", ["Car Rentals", "Bike Rentals", "E-scooter Rentals", "RV or Campervan Rentals"])
        public_transport_preferences = st.multiselect("Select public transport preferences:", ["Urban Travel", "Long-Distance Travel", "Eco-conscious options", "Local Experiences"])
        travel_party = st.selectbox("Select your travel party:", ["Solo", "Couple", "Family", "Friends"])

        # Button
        submit3 = st.button("Find Transport Options!")
        if submit3:
            # Construct the input plan based on user inputs
            input_transport = f"""
            Location: {location}
            Mode of Transport: {mode_of_transport}
            Rental Services: {', '.join(rental_services)}
            Public Transport Preferences: {', '.join(public_transport_preferences)}
            Travel Party: {travel_party}
            """
        
            response = get_response(input_prompt_transport, input_transport)
            st.subheader("Transport Recommendations: ")
            # Justify the response text using HTML and CSS
            justified_response = f'<div style="text-align: justify;">{response}</div>'
            st.markdown(justified_response, unsafe_allow_html=True)


    # If the choice is Food Preferences
    if section_choice == "Food Preferences":

        # Food Preferences Prompt Template
        input_prompt_food = """
        "You are an expert Travel and Culinary Advisor. Your job is to provide personalized transport and food recommendations based on the user's preferences.
        Key Input Considerations:
        1. Location:
            Take into account the specified destination to tailor your recommendations to the local culture, availability of services, and unique culinary experiences.
        2. Dietary Restrictions:
            Address the user's dietary needs such as vegan, vegetarian, halal, gluten-free, or No Restrictions options. Ensure all recommendations respect these restrictions without compromising on taste and variety.
        3. Interest in Local Cuisine:
            Explore the user’s interest in local culinary experiences, such as trying authentic dishes, or enjoying street food. Incorporate activities and dining options that align with these interests.
        4. Dining Experience:
            Provide suggestions that match the user's preferred dining style, whether it’s fine dining for a sophisticated evening, casual dining for relaxed meals, or street vendors for quick and authentic local bites.
        5. Ambiance Preferences:
            Include ambiance preferences, such as romantic settings for couples, family-friendly environments for larger groups, or trendy spots for social gatherings.
        6. Cuisine Variety:
            Recommend restaurants and eateries offering a wide range of cuisines, including Italian, Thai, Indian, fusion dishes, or local specialties.
        Return the response using markdown.
        """

        # Input Fields
        location = st.text_input("Enter the location for transport recommendations:")
        dietary_restrictions = st.multiselect("Select dietary restrictions:", ["Vegan", "Vegetarian", "Halal", "Gluten-Free", "No Restrictions"])
        interest_in_local_cuisine = st.multiselect("Select interests in local cuisine:", ["Exploring Authentic Flavors", "Street Food"])
        dining_experience = st.selectbox("Select type of dining experience:", ["Fine Dining", "Casual Dining", "Street Vendors"])
        ambiance_preferences = st.selectbox("Select ambiance preferences:", ["Romantic", "Family-Friendly", "Trendy"])
        cuisine_variety = st.multiselect("Select preferred cuisines:", ["Italian", "Thai", "Indian", "Fusion", "Local Specialties"])

        # Button
        submit4 = st.button("Find Food Options!")
        if submit4:
            # Construct the input plan based on user inputs
            input_food = f"""
            Location: {location}
            Dietary Restrictions: {', '.join(dietary_restrictions)}
            Interest in Local Cuisine: {', '.join(interest_in_local_cuisine)}
            Dining Experience: {dining_experience}
            Ambiance Preferences: {ambiance_preferences}
            Cuisine Variety: {', '.join(cuisine_variety)}
            """
        
            response = get_response(input_prompt_food, input_food)
            st.subheader("Food Recommendations: ")
            # Justify the response text using HTML and CSS
            justified_response = f'<div style="text-align: justify;">{response}</div>'
            st.markdown(justified_response, unsafe_allow_html=True)
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_age_gender_model():
    """
    Create a CNN model for predicting age and gender from facial images.
    Returns two separate models: one for age prediction and one for gender prediction.
    """
    # Base model architecture
    def create_base_model(input_shape=(200, 200, 3)):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
        ])
        return model
    
    # Age prediction model
    age_model = create_base_model()
    age_model.add(Dense(1, activation='linear', name='age_output'))
    age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Gender prediction model
    gender_model = create_base_model()
    gender_model.add(Dense(1, activation='sigmoid', name='gender_output'))
    gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return age_model, gender_model

def preprocess_image(image_path):
    """
    Preprocess an image for the age and gender prediction models.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        raise ValueError("No faces detected in the image")
    
    # Extract face
    x, y, w, h = faces[0]  # Take the first face
    face = img[y:y+h, x:x+w]
    
    # Resize and normalize
    face = cv2.resize(face, (200, 200))
    face = face / 255.0  # Normalize to [0,1]
    
    return face, img, (x, y, w, h)

def get_diet_recommendations(age, gender):
    """
    Generate healthy diet recommendations based on age and gender.
    
    Args:
        age (int): Predicted age
        gender (str): Predicted gender ('male' or 'female')
    
    Returns:
        dict: Diet recommendations including calorie intake and food suggestions
    """
    # Age groups
    if age < 13:
        age_group = "child"
    elif 13 <= age < 19:
        age_group = "teen"
    elif 19 <= age < 31:
        age_group = "young_adult"
    elif 31 <= age < 51:
        age_group = "adult"
    elif 51 <= age < 65:
        age_group = "middle_age"
    else:
        age_group = "senior"
    
    # Base calorie recommendations by age group and gender
    # These are approximations and should be adjusted based on activity level,
    # weight, height, and other factors
    calorie_needs = {
        "child": {"male": 1800, "female": 1600},
        "teen": {"male": 2200, "female": 1800},
        "young_adult": {"male": 2500, "female": 2000},
        "adult": {"male": 2400, "female": 1900},
        "middle_age": {"male": 2200, "female": 1800},
        "senior": {"male": 2000, "female": 1600}
    }
    
    # Specific nutritional recommendations
    recommendations = {
        "child": {
            "protein": "Essential for growth. Include lean meats, dairy, beans, and eggs.",
            "calcium": "Critical for bone development. Focus on dairy products and fortified foods.",
            "fruits_vegetables": "Aim for 5 servings daily for essential vitamins and minerals.",
            "avoid": "Limit processed sugars and high-sodium foods."
        },
        "teen": {
            "protein": "Important for muscle development. Include a variety of sources.",
            "calcium": "Critical for bone density. Include dairy or fortified alternatives.",
            "iron": "Especially important for teenage girls. Include lean red meat, beans, and leafy greens.",
            "avoid": "Minimize fast food, sugary drinks, and highly processed snacks."
        },
        "young_adult": {
            "protein": "0.8g per kg of body weight daily from diverse sources.",
            "fiber": "25-30g daily from whole grains, fruits, and vegetables.",
            "healthy_fats": "Include sources of omega-3s like fish, nuts, and seeds.",
            "avoid": "Limit alcohol and high-sodium processed foods."
        },
        "adult": {
            "balanced_diet": "Focus on nutrient-dense whole foods.",
            "portion_control": "Be mindful of portion sizes to maintain healthy weight.",
            "antioxidants": "Include colorful fruits and vegetables.",
            "avoid": "Limit saturated fats and added sugars."
        },
        "middle_age": {
            "protein": "Slightly increased needs to prevent muscle loss.",
            "calcium": "Important for bone health, especially for women after menopause.",
            "fiber": "Critical for digestive health and cholesterol management.",
            "avoid": "Reduce sodium intake to support heart health."
        },
        "senior": {
            "protein": "Higher needs (1-1.2g per kg) to prevent muscle loss.",
            "vitamin_D": "Often requires supplementation in addition to dietary sources.",
            "B_vitamins": "Important for energy and cognitive function.",
            "hydration": "Ensure adequate fluid intake throughout the day.",
            "avoid": "Very low-calorie diets that might lead to nutrient deficiencies."
        }
    }
    
    # Gender-specific recommendations
    gender_specific = {
        "male": {
            "teen": "Higher protein and calorie needs to support growth spurts.",
            "adult": "Focus on heart-healthy foods like fatty fish, nuts, and whole grains.",
            "middle_age": "Monitor saturated fat intake to support heart health.",
            "senior": "Maintain adequate protein intake to preserve muscle mass."
        },
        "female": {
            "teen": "Ensure adequate iron intake to replace losses from menstruation.",
            "young_adult": "Consider folate-rich foods if planning pregnancy.",
            "adult": "Include sources of calcium and vitamin D for bone health.",
            "middle_age": "Consider foods rich in phytoestrogens during menopause.",
            "senior": "Focus on calcium and vitamin D for bone health."
        }
    }
    
    # Sample meal plans
    meal_plans = {
        "breakfast": {
            "child": "Whole grain cereal with milk and fresh fruit",
            "teen": "Greek yogurt with berries and granola",
            "young_adult": "Avocado toast with poached eggs",
            "adult": "Oatmeal with nuts, seeds, and fresh fruit",
            "middle_age": "Smoothie with protein, greens, and berries",
            "senior": "Whole grain toast with egg and fruit"
        },
        "lunch": {
            "child": "Turkey sandwich on whole grain bread with carrot sticks",
            "teen": "Chicken wrap with vegetables and hummus",
            "young_adult": "Grain bowl with lean protein and roasted vegetables",
            "adult": "Large salad with lean protein and olive oil dressing",
            "middle_age": "Soup and salad with lean protein",
            "senior": "Baked fish with steamed vegetables and quinoa"
        },
        "dinner": {
            "child": "Baked chicken, sweet potato, and broccoli",
            "teen": "Pasta with lean protein and vegetables",
            "young_adult": "Stir-fry with lean protein and plenty of vegetables",
            "adult": "Grilled fish with roasted vegetables and whole grains",
            "middle_age": "Lean protein with half plate of vegetables and quarter plate of whole grains",
            "senior": "Vegetable soup with beans and whole grain bread"
        }
    }
    
    # Compile recommendations
    result = {
        "estimated_age": age,
        "gender": gender,
        "age_group": age_group,
        "daily_calorie_needs": calorie_needs[age_group][gender],
        "nutritional_focus": recommendations[age_group],
        "gender_specific_advice": gender_specific[gender].get(age_group, "No specific recommendations"),
        "sample_meals": {
            "breakfast": meal_plans["breakfast"][age_group],
            "lunch": meal_plans["lunch"][age_group],
            "dinner": meal_plans["dinner"][age_group]
        }
    }
    
    return result

def predict_and_recommend(image_path, age_model, gender_model):
    """
    Predict age and gender from an image and generate diet recommendations.
    
    Args:
        image_path (str): Path to the image
        age_model: Model for age prediction
        gender_model: Model for gender prediction
    
    Returns:
        tuple: Processed image with annotations, diet recommendations
    """
    try:
        # Preprocess image
        face, original_img, (x, y, w, h) = preprocess_image(image_path)
        face_batch = np.expand_dims(face, axis=0)
        
        # Predict age and gender
        predicted_age = float(age_model.predict(face_batch)[0][0])
        gender_prob = float(gender_model.predict(face_batch)[0][0])
        predicted_gender = "male" if gender_prob < 0.5 else "female"
        
        # Get diet recommendations
        recommendations = get_diet_recommendations(int(predicted_age), predicted_gender)
        
        # Annotate image
        cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"Age: {int(predicted_age)}, Gender: {predicted_gender}"
        cv2.putText(original_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return original_img, recommendations
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    """
    Main function to demonstrate the system.
    """
    print("Creating age and gender prediction models...")
    age_model, gender_model = create_age_gender_model()
    
    print("Note: In a real application, these models would be trained on appropriate datasets.")
    print("For a fully functional system, you would need to:")
    print("1. Train the models on age and gender labeled facial datasets")
    print("2. Save and load the trained models")
    print("3. Create a user interface for image upload")
    
    print("\nTo use this system with pretrained models, you would:")
    print("1. Replace the model creation with loading pretrained models")
    print("2. Use the predict_and_recommend function with an image path")
    print("3. Display the annotated image and diet recommendations")

if __name__ == "__main__":
    main()
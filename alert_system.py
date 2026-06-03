from dotenv import load_dotenv
import os

load_dotenv()

import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
import os
from twilio.rest import Client
import time

# --- CONFIGURATION ---
# !! IMPORTANT: For production, use environment variables, not hardcoded keys!
# os.environ.get('CLOUDINARY_NAME')
# os.environ.get('CLOUDINARY_KEY')
# os.environ.get('CLOUDINARY_SECRET')
# os.environ.get('TWILIO_SID')
# os.environ.get('TWILIO_TOKEN')
# os.environ.get('TWILIO_FROM_NUMBER')
# os.environ.get('YOUR_WHATSAPP_NUMBER')

CLOUDINARY_NAME = os.getenv("CLOUDINARY_NAME")
CLOUDINARY_KEY = os.getenv("CLOUDINARY_KEY")
CLOUDINARY_SECRET = os.getenv("CLOUDINARY_SECRET")

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

# --- !!! ENTER YOUR NUMBER HERE !!! ---
# Must be in format: "whatsapp:+[CountryCode][PhoneNumber]"
# Example for India: "whatsapp:+919876543210"
# Example for US: "whatsapp:+12025550147"
YOUR_WHATSAPP_NUMBER = os.getenv("YOUR_WHATSAPP_NUMBER")

twilio_client = None

def configure_alerts():
    """Initializes the Cloudinary and Twilio clients."""
    global twilio_client
    try:
        # Configure Cloudinary
        cloudinary.config(
            cloud_name=CLOUDINARY_NAME,
            api_key=CLOUDINARY_KEY,
            api_secret=CLOUDINARY_SECRET,
            secure=True
        )
        
        # Configure Twilio
        twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
        
        print("Alert system configured (Cloudinary & Twilio).")
        
        if YOUR_WHATSAPP_NUMBER == "whatsapp:+91xxxxxxxxxx":
            print("\n" + "="*50)
            print("WARNING: Please update YOUR_WHATSAPP_NUMBER in alert_system.py")
            print("="*50 + "\n")
            
    except Exception as e:
        print(f"Error configuring alert system: {e}")

def upload_to_cloudinary(frame):
    """Encodes and uploads a CV2 frame to Cloudinary."""
    try:
        # Encode the frame as JPG
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Upload to Cloudinary
        # We use a timestamp as the public_id to avoid overwriting files
        public_id = f"drowsy_alert_{int(time.time())}"
        
        upload_result = cloudinary.uploader.upload(
            buffer.tobytes(),
            public_id=public_id,
            overwrite=True,
            resource_type="image"
        )
        
        secure_url = upload_result.get('secure_url')
        if secure_url:
            print(f"Image uploaded to Cloudinary: {secure_url}")
            return secure_url
        else:
            print(f"Cloudinary upload failed: {upload_result}")
            return None
            
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        return None

def send_whatsapp_alert(image_url):
    """Sends a WhatsApp message with the image URL via Twilio."""
    global twilio_client
    if not twilio_client:
        print("Twilio client not initialized. Call configure_alerts() first.")
        return

    if YOUR_WHATSAPP_NUMBER == "whatsapp:+91xxxxxxxxxx":
        print("Cannot send alert: YOUR_WHATSAPP_NUMBER is not set.")
        return
        
    try:
        message_body = "DROWSINESS ALERT! \n\nPotential drowsy driver detected."
        
        message = twilio_client.messages.create(
            from_=TWILIO_FROM_NUMBER,
            body=message_body,
            media_url=[image_url],
            to=YOUR_WHATSAPP_NUMBER
        )
        
        print(f"WhatsApp alert sent successfully to {YOUR_WHATSAPP_NUMBER}. SID: {message.sid}")
        
    except Exception as e:
        print(f"Error sending Twilio message: {e}")

# Example of how to use this file (for testing)
if __name__ == "__main__":
    print("Testing alert system...")
    configure_alerts()
    
    # Create a dummy image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TEST ALERT", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    image_url = upload_to_cloudinary(dummy_frame)
    
    if image_url:
        print("Test upload successful. Sending test WhatsApp message...")
        send_whatsapp_alert(image_url)
    else:
        print("Test upload failed.")

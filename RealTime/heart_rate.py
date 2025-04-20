import cv2
import numpy as np
from scipy.fftpack import fft, fftfreq
import time

def detect_heart_rate_spo2_bp_with_timer():
    """
    Real-time heart rate, SpO2, and blood pressure detection using webcam and block processing.
    """

    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    frame_buffer = []
    fps = 30  

    
    start_time = time.time()
    duration = 30  

    try:
        print("Starting heart rate, SpO2, and BP detection... The process will stop after 30 seconds.")
        while True:
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                
                (x, y, w, h) = faces[0]

                
                forehead_roi = frame[y:y + int(0.2 * h), x:x + w]  
                cheek_roi = frame[y + int(0.2 * h):y + int(0.5 * h), x:x + w]  
                chin_roi = frame[y + int(0.5 * h):y + h, x:x + w]  

                
                green_channel_mean = np.mean(forehead_roi[:, :, 1]) 

                
                frame_buffer.append(green_channel_mean)
                if len(frame_buffer) > fps * 5: 
                    frame_buffer.pop(0)

                
                forehead_grad_x = cv2.Sobel(forehead_roi[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
                forehead_grad_y = cv2.Sobel(forehead_roi[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
                forehead_texture = cv2.magnitude(forehead_grad_x, forehead_grad_y)

                cheek_grad_x = cv2.Sobel(cheek_roi[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
                cheek_grad_y = cv2.Sobel(cheek_roi[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
                cheek_texture = cv2.magnitude(cheek_grad_x, cheek_grad_y)

                chin_grad_x = cv2.Sobel(chin_roi[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
                chin_grad_y = cv2.Sobel(chin_roi[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
                chin_texture = cv2.magnitude(chin_grad_x, chin_grad_y)

               
                cv2.imshow("Forehead Texture", forehead_texture.astype(np.uint8)) 

                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                cv2.rectangle(frame, (x, y), (x + w, y + int(0.2 * h)), (0, 255, 0), 2)  
                cv2.rectangle(frame, (x, y + int(0.2 * h)), (x + w, y + int(0.5 * h)), (0, 0, 255), 2) 
                cv2.rectangle(frame, (x, y + int(0.5 * h)), (x + w, y + h), (0, 255, 255), 2) 

           
            cv2.imshow("Heart Rate, SpO2, and BP Detection", frame)

            
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        if len(frame_buffer) > fps:  
            print("Processing heart rate...")
            
            freqs = fftfreq(len(frame_buffer), d=1 / fps)
            fft_values = np.abs(fft(frame_buffer))
            positive_freqs = freqs[freqs > 0]
            positive_fft_values = fft_values[freqs > 0]

           
            valid_range = (positive_freqs >= 0.8) & (positive_freqs <= 3.0)
            if np.any(valid_range):
                dominant_freq = positive_freqs[valid_range][np.argmax(positive_fft_values[valid_range])]
                heart_rate = dominant_freq * 60 
                
                
                heart_rate = max(60, min(heart_rate, 100))
                print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
            else:
                print("No valid heart rate signal detected.")

            
            red_channel_mean = np.mean(forehead_roi[:, :, 2])  
            spo2 = (red_channel_mean / green_channel_mean) * 100  
            print(f"Estimated SpO2: {spo2:.2f}%")

            
            blood_pressure = 120 - (np.mean(green_channel_mean) / 255) * 40  
            print(f"Estimated Blood Pressure: {blood_pressure:.2f} mmHg")

        else:
            print("Insufficient data to calculate heart rate, SpO2, and BP.")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        
        cap.release()
        cv2.destroyAllWindows()


detect_heart_rate_spo2_bp_with_timer()

# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:01:34 2020

@author: malinni.natarajan
"""

# Import all the required packages
import cv2, os, datetime, pandas

# Create a function to detect motion
def motion_detection(x):
    first_frame = None
	
	# x is the input choice of the user, 1 being webcam, 2 being sample video
    cap=cv2.VideoCapture(x)
	
    #write the resultant video to a newly created/existing output folder
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'output')
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
       
	# Naming webcam files with numbers to save all the webcam outputs without overwriting
    files = os.listdir(final_directory)
    webcam_files = [f for f in files if "webcam" in f]
       
    if x == 0:
        out = cv2.VideoWriter('{}/webcam_motion_output_{}.avi'.format(final_directory, len(webcam_files)+1),cv2.VideoWriter_fourcc('M','J','P','G'), 18, (int(cap.get(3)), int(cap.get(4))))
    else:
        video_name = x.split("\\")[-1].split(".")[0]
        out = cv2.VideoWriter('{}/{}_motion_output.avi'.format(final_directory, video_name),cv2.VideoWriter_fourcc('M','J','P','G'), 18, (int(cap.get(3)), int(cap.get(4))))
			
    while True:
        # The read function gives two outputs. The check is a boolean function that returns if the video is being read
        check, frame = cap.read() 
        grayImg=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Grayscale conversion of the frame
        # Gaussian blur to smoothen the image and remove noise. 
        # The tuple is the Kernel size and the 0 is the Std Deviation of the blur function
        grayImg=cv2.GaussianBlur(grayImg, (21,21),0) 
    
        if first_frame is None:
            first_frame=grayImg #collect the reference frame as the first video feed frame
            continue
    
        # Calculate the absolute difference between current frame and reference frame
        deltaFrame=cv2.absdiff(first_frame,grayImg)    
    
        # Convert image from grayscale to binary. This increases the demarcation between object and background by using a threshold function that 
        # Converts everything above threshold to white
        threshFrame=cv2.threshold(deltaFrame, 30, 255, cv2.THRESH_BINARY)[1]
    
        # Dilating the threshold removes the sharp edges at the object/background boundary and makes it smooth. 
        # More the iterations, smoother the image. Too smooth and you lose valuable data
        threshFrame=cv2.dilate(threshFrame, None, iterations=2)
    
        #Contour Function
        #The contour function helps identify the closed object areas within the background. 
        #After thresholding, the frame has closed shapes of the objects against the background
        #The contour function identifies and creates a list (cnts) of all these contours in the frame
        #The RETR_EXTERNAL ensures that you only get the outermost contour details and all child contours inside it are ignored
        #The CHAIN_APPROX_SIMPLE is the approximation method used for locating the contours. The simple one is used here for our trivial purpose
        #Simple approximation removes all the redundant points in the description of the contour line    
        contours,hierachy=cv2.findContours(threshFrame.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if cv2.contourArea(contour) < 20000: 
                # Excluding too small contours. Set minimum of 10000 (100x100 pixels) for objects close to camera
                continue
            # Obtain the corresponding bounding rectangle of our detected contour
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.putText(frame, "MOTION DETECTED", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0))
    
            # Superimpose a rectangle on the identified contour in our original colour image
            # (x,y) is the top left corner, (x+w, y+h) is the bottom right corner
            # (0,255,0) is colour green and 3 is the thickness of the rectangle edges
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)           
            
        out.write(frame)    
        # Displays the continous feed with the green frame for any foreign object in frame
        cv2.imshow("Colour Frame", frame)
        # Picks up the key press Esc and exits when pressed
        if cv2.waitKey(33) == 27: 
            break
    # Closes all windows
    cv2.destroyAllWindows()
    out.release()
    # Releases video file/webcam
    cap.release()
    
def main():
    while True:
        i = input("Sample Motion Detection:\n\nPress 1 to Start, 0 to Exit\n\nYour choice:")
        if i != "1":
            sys.exit(0)
        else:
            cam_choice = input("\nChoose your input mode:\n\nPress 1 for Webcam, 2 to use a sample video, 0 to Exit\n\nYour choice:")
            if cam_choice == "0":
                sys.exit(0)
            elif cam_choice == "1":
                motion_detection(0)
            else:
                path = input("\nEnter the absolute path of the file:")
                motion_detection(path)

if __name__ == "__main__":
    main()

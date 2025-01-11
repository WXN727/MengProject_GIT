import tkinter as tk
import subprocess
import threading

process_facial = None  # Global variable to hold the process reference for facial.py
process_pose_main = None  # Global variable to hold the process reference for PoseMain.py


def execute_script():
    global process_facial, process_pose_main

    def run_pose_main_script():
        global process_pose_main
        process_pose_main = subprocess.Popen(["python", "PoseMain.py"])
        process_pose_main.communicate()

    def run_facial_script():
        global process_facial
        process_facial = subprocess.Popen(["python", "Face_old.py"])
        process_facial.communicate()

    # Start both scripts in separate threads
    facial_thread = threading.Thread(target=run_facial_script)
    pose_main_thread = threading.Thread(target=run_pose_main_script)

    pose_main_thread.start()
    facial_thread.start()


def end_script():
    global process_facial, process_pose_main

    # Terminate the facial.py process
    if process_facial and process_facial.poll() is None:  # Check if the process is still running
        process_facial.terminate()  # Terminate the process
        process_facial = None  # Reset the process reference

    # Terminate the PoseMain.py process
    if process_pose_main and process_pose_main.poll() is None:  # Check if the process is still running
        process_pose_main.terminate()  # Terminate the process
        process_pose_main = None  # Reset the process reference


# Create the main window
root = tk.Tk()
root.title("Execute YOLO5 Script GUI")
root.geometry("1600x800")  # Set the window size to 800x400 pixels

button_font = ("Helvetica", 20)  # Define the font and size

# Create a button to execute the YOLO5 script
execute_button = tk.Button(root, text="Execute YOLO5 Script", command=execute_script, width=50, height=5,font=button_font)
execute_button.pack(pady=20)

# Create an "End" button to terminate the YOLO5 script
end_button = tk.Button(root, text="End Script", command=end_script, width=50, height=5,font=button_font)
end_button.pack(pady=20)

# Run the main loop
root.mainloop()

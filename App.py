
import re
#import anthropic need to be solved
import streamlit as st
import subprocess
import os
import pandas as pd
from PIL import Image
from pytube import YouTube
import subprocess
from pydub import AudioSegment
import speech_recognition as sr
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize
import random
import google.generativeai as genai
import csv
import base64



def display_article_content(csv_file_path, summarize_response):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Creating a placeholder for the summary at the bottom
    summary_placeholder = st.empty()
    
    # Button at the top to jump to the overview (summary)
    

    # Check if the jump to summary button was pressed and set the view accordingly
    if 'jump_to_summary' in st.session_state and st.session_state.jump_to_summary:
        st.session_state.jump_to_summary = False  # Resetting the state
        summary_placeholder.markdown(f"**Summary/Overview:** {summarize_response}")  # Display summary
        # Optionally, you could add a button or link to jump back to the top
        return  # Exit the function early

    # Iterate over each row in the DataFrame and display its content
    for _, row in df.iterrows():
        timestamp, image_path, text, response = row['Timestamp'], row['Screenshot'], row['Text'], row['Response']

        with st.container():
            try:
                image = Image.open(image_path)
                st.image(image, caption="Event Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")

            st.markdown(f"**Part ({timestamp}):** {text}")  # Bold the timestamp for emphasis
            st.markdown("---")  # Separator for visual distinction
            st.markdown(f"*{response}*")  # Italicize the response for differentiation

    # After iterating through all items, display the summary at the designated placeholder
    summary_placeholder.markdown(f"**Summary/Overview:** {summarize_response}")

def summarize(csv_path):
    df = pd.read_csv(csv_path)
    # Extract all strings from the 'Response' column and combine them into one long string separated by commas
    combined_responses = ', '.join(df['Response'].astype(str))
    text = "summarize the next text, avoid been over specific, try to explain the main features (bulletpoint are great use) :" + combined_responses
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(text, stream=True)
    response.resolve()
    return response.text

def summarize_claude(csv_path,model,client):
    df = pd.read_csv(csv_path)
    # Extract all strings from the 'Response' column and combine them into one long string separated by commas
    combined_responses = ', '.join(df['Response'].astype(str))
    text = "summarize the next text, avoid been over specific, try to explain the main features (bulletpoint are great use) :" + combined_responses
    message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": text}
    ]
    )
    return (message.content[0].text)
        

# Function to generate report
def generate_report(input_file,model,client):
    # Process the video
    print(model)
    input_name = os.path.basename(input_file)
    folder_path = 'C:/Users/orile/Documents/TutorialDigest/'
    input_name = input_name[0:-4]
    # Step 1: Extract Audio
    audio_output = extract_audio_ffmpeg(input_file)
    
    # Step 2: Transcribe Audio
    transcribe_audio(audio_output)

    # Step 3: Extract Screenshots
    screenshot_folder = f"{folder_path}{input_name}_Screenshots"
    extract_screenshots_ffmpeg(input_file, screenshot_folder)

    # Step 4: Delete Similar Following Frames
    delete_similar_following_frames(screenshot_folder)

    # Step 5: Select Random Screenshot and Update CSV 
    csv_file_path = f"{folder_path}{input_name}_audio_transcriptions.csv"
    output_csv_file_path = f"{folder_path}{input_name}_updated_transcriptions.csv"
    select_random_screenshot_and_update_csv(screenshot_folder, csv_file_path, output_csv_file_path)
    #select_random_screenshot_and_update_csv_claude(screenshot_folder, csv_file_path, output_csv_file_path,model,client)
    # Step 5.5 - summerize everything
    sum = summarize(output_csv_file_path)
    #sum = summarize_claude(output_csv_file_path,model,client)
    # Step 6: Convert CSV to HTML

    #csv_to_html(output_csv_file_path, html_file_path, screenshot_folder)
    # Load the HTML content from output.html
    display_article_content(output_csv_file_path,sum)
    
def sanitize_filename(filename, max_length=20):
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove illegal characters
    filename = re.sub(r'[\\/*?:"<>|]', '', filename)
    # Truncate to the max_length
    return filename[:max_length]

def download_youtube_video(url, path):
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        
        # Sanitize the title
        sanitized_title = f"{sanitize_filename(yt.title)}.mp4"
        
        # Download the video
        stream.download(output_path=path, filename=sanitized_title)
        
        print(f"Downloaded '{yt.title}' successfully.")
        
        # Construct the full path for the downloaded file
        downloaded_file_path = f"{path}/{sanitized_title}"
        return downloaded_file_path
    except Exception as e:
        print(f"Error downloading video: {e}")
    
# Function to extract audio using FFmpeg
def extract_audio_ffmpeg(input_file):
        print("Extracting audio...",end='\r')
        # Construct the FFmpeg command
        name = os.path.basename(input_file)
        output_path = f'{name[0:-4]}_audio.wav'
        command = ['ffmpeg', '-i', input_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', output_path]
        # Execute the command
        subprocess.run(command, check=True)
        print("Extracting audio complete")
        return output_path

    # Function to transcribe audio using SpeechRecognition
def transcribe_audio(input_wav, interval=15000):
        print("Transcription process started")
        audio = AudioSegment.from_wav(input_wav)
        segments = [audio[i:i+interval] for i in range(0, len(audio), interval)]
        recognizer = sr.Recognizer()
        name = os.path.basename(input_wav)
        with open(f'{name[0:-4]}_transcriptions.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Text'])

            for index, segment in enumerate(segments):
                segment_file = f"temp_segment_{index}.wav"
                segment.export(segment_file, format="wav")
                
                with sr.AudioFile(segment_file) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                    except sr.UnknownValueError:
                        text = "Audio not understood"
                    except sr.RequestError as e:
                        text = f"Could not request results; {e}"

                start_time = str(index * 15)  # Starting time of segment in seconds
                end_time = str((index + 1) * 15)  # Ending time of segment in seconds
                timestamp = f"{start_time} - {end_time}"
                writer.writerow([timestamp, text])
                print(f"Processed the segment:'{timestamp}' out of the audio ({index+1} out of {len(segments)})", end='\r')
                
                # Delete the temporary file after it's been transcribed
                os.remove(segment_file)
                
        print("\nTranscription process complete - open transcriptions.csv for results")

    # Function to extract screenshots using FFmpeg
def extract_screenshots_ffmpeg(video_path, output_folder):
        print("Extracting screenshots...",end='\r')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=1',
            os.path.join(output_folder, 'screenshot_%d.png')
        ]
        
        subprocess.run(command, check=True)
        print("Extracting screenshots complete")

    # Function to compare images and delete similar frames
def compare_images(img_path1, img_path2):
        # Load the images
        img1 = imread(img_path1, as_gray=True)
        img2 = imread(img_path2, as_gray=True)
        
        # Resize images to match (if they're not the same size)
        img2 = resize(img2, img1.shape, anti_aliasing=True, preserve_range=True)
        
        # Compute SSIM between the two images
        # Specify the data range for 8-bit images, which is 255
        similarity_index = ssim(img1, img2, data_range=img1.max() - img1.min())
        return similarity_index

    # Function to delete similar following frames
def delete_similar_following_frames(folder_path, similarity_threshold=0.9):
        # List all png images in the folder
        print("Cleaning up similar frames...")
        images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
        total_images, deleted = len(images), 0
        i = 0
        while i < len(images)-1:
            # Compare current image with the next one
            sim_index = compare_images(images[i], images[i+1])
            
            if sim_index > similarity_threshold:
                print(f"{os.path.basename(images[i+1])} deleted due to similarity")
                deleted += 1
                os.remove(images[i+1])
                images.pop(i+1)  # Remove the path of the deleted image from the list
            else:
                i += 1  # Move to the next image only if the current one wasn't deleted
            print(f"out of {total_images} images - {deleted} deleted, {total_images-deleted} left",end='\r')        
        print(f"\nCleanup complete: {deleted}/{total_images} frames removed.")

    # Function to select random screenshot and update CSV
def select_random_screenshot_and_update_csv(folder_path, csv_file_path, output_csv_file_path):
        print("Updating CSV with screenshots and responses...")
        screenshot_files = [f for f in os.listdir(folder_path) if f.startswith('screenshot_') and f.endswith('.png')]
        screenshot_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
        
        with open(csv_file_path, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            transcript_entries = list(reader)[1:]  # Skip header

        # Prepare to write to the new CSV file
        with open(output_csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header for the new CSV file
            writer.writerow(['Timestamp', 'Screenshot', 'Text', 'Response'])
            prompt = "Take the next given image and text and super analyze and extract as much information as possible:"
            for i in range(len(transcript_entries)):
                # Select a random screenshot within the interval
                start_index = i * 15 + 1  # +1 to match file naming convention
                end_index = min(start_index + 15, len(screenshot_files) + 1)
                selected_screenshot_indices = list(range(start_index, end_index))
                if not selected_screenshot_indices:
                    continue
                selected_screenshot_index = random.choice(selected_screenshot_indices)
                selected_screenshot = f'{folder_path}/screenshot_{selected_screenshot_index}.png'
                
                if i < len(transcript_entries):
                    text = prompt + transcript_entries[i][1]  # Assuming text is in the second column
                    timestamp = transcript_entries[i][0]
                else:
                    text = "No corresponding text found."
                    timestamp = f"{i*15}-{(i+1)*15}"

                # Assuming the use of a fictional genai library for text generation
                img_path = os.path.join(folder_path, selected_screenshot)
                img = Image.open(img_path)
                model = genai.GenerativeModel('gemini-pro-vision')
                response = model.generate_content([text, img], stream=True)
                response.resolve()
                generated_response = response.text  # Placeholder for actual response

                writer.writerow([timestamp, selected_screenshot, transcript_entries[i][1], generated_response])
                print(f"Processed interval: {timestamp}, selected screenshot: {selected_screenshot}")
        print("CSV update complete.")

def select_random_screenshot_and_update_csv_claude(folder_path, csv_file_path, output_csv_file_path,model,client):
        print("Updating CSV with screenshots and responses...")
        screenshot_files = [f for f in os.listdir(folder_path) if f.startswith('screenshot_') and f.endswith('.png')]
        screenshot_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
        
        with open(csv_file_path, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            transcript_entries = list(reader)[1:]  # Skip header

        # Prepare to write to the new CSV file
        with open(output_csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header for the new CSV file
            writer.writerow(['Timestamp', 'Screenshot', 'Text', 'Response'])
            prompt = "Take the next given image and text and super analyze and extract as much information as possible:"
            for i in range(len(transcript_entries)):
                # Select a random screenshot within the interval
                start_index = i * 15 + 1  # +1 to match file naming convention
                end_index = min(start_index + 15, len(screenshot_files) + 1)
                selected_screenshot_indices = list(range(start_index, end_index))
                if not selected_screenshot_indices:
                    continue
                selected_screenshot_index = random.choice(selected_screenshot_indices)
                selected_screenshot = f'{folder_path}/screenshot_{selected_screenshot_index}.png'
                
                if i < len(transcript_entries):
                    text = prompt + transcript_entries[i][1]  # Assuming text is in the second column
                    timestamp = transcript_entries[i][0]
                else:
                    text = "No corresponding text found."
                    timestamp = f"{i*15}-{(i+1)*15}"

                # Assuming the use of a fictional genai library for text generation
                img_path = os.path.join(folder_path, selected_screenshot)
                with open(img_path,"rb") as image_file:
                     image1_data = image_file.read()
                image1_media_type = "image/png"
                image1_data = base64.b64encode(image1_data).decode("utf-8")
                message = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image1_media_type,
                                        "data": image1_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": text
                                }
                            ],
                        }
                    ],
                )  

                generated_response = message  # Placeholder for actual response

                writer.writerow([timestamp, selected_screenshot, transcript_entries[i][1], generated_response])
                print(f"Processed interval: {timestamp}, selected screenshot: {selected_screenshot}")
        print("CSV update complete.")

    # Function to convert CSV to HTML
def csv_to_html(csv_file_path, html_file_path, screenshot_folder_path):
        print("Generating HTML from CSV...")
        html_output = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Transcription and Responses</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .entry { margin-bottom: 40px; }
                .timestamp { font-weight: bold; }
                .image { margin-top: 10px; margin-bottom: 10px; }
                .text, .response { margin-top: 5px; }
            </style>
        </head>
        <body>
        """

        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    screenshot_path = os.path.join(screenshot_folder_path, row['Screenshot'])
                    # Attempt to open and encode the screenshot
                    with open(screenshot_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode()
                except KeyError:
                    print(f"KeyError: 'Screenshot' column not found in the CSV.")
                    encoded_string = ""
                except FileNotFoundError:
                    print(f"FileNotFoundError: {screenshot_path} not found.")
                    encoded_string = ""
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    encoded_string = ""
                
                html_output += f"""
                <div class="entry">
                    <div class="timestamp">{row.get('Timestamp', 'Unknown Timestamp')}: {row.get('Text', 'No text available')}</div>
                    <div class="image"><img src="data:image/png;base64,{encoded_string}" alt="Screenshot missing" width="400"></div>
                    <div class="response">{row.get('Response', 'No response available')}</div>
                </div>
                """
        
        html_output += """
        </body>
        </html>
        """
        
        with open(html_file_path, 'w', encoding='utf-8') as html_file:
            html_file.write(html_output)
        print("HTML generation complete.")
# Main Streamlit code
def main():
    st.title("Video To Report Generation App")
    
    # Sidebar for token input, file upload, and YouTube URL
    st.sidebar.title("To Use This App, You Need To Enter Claude API Key\nUpload an MP4 video or enter a YouTube URL\nShould not be longer than 5 mins")
    
    # Token is required to proceed
    token = st.sidebar.text_input("Enter Claude API Key", type="password", help="This field is required.")
    client = "need to be fixed"
    """client = anthropic.Anthropic(
    api_key=token
    )"""
    # Model selection dropdown
    model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]  # Add your actual model names/options here
    selected_model = st.sidebar.selectbox("Choose a Model", options=model_options)
    if selected_model == model_options[0]:
         st.sidebar.text("Claude Opus - Most powerful model\nfor highly complex tasks")
    elif selected_model == model_options[1]:
         st.sidebar.text("Claude Sonnet - Ideal balance of\nintelligence and speed for\nenterprise workloads")
    elif selected_model == model_options[2]:
         st.sidebar.text("Claude Haiku - Fastest and most \ncompact model for \nnear-instant responsiveness")     
    
    uploaded_file = st.sidebar.file_uploader("Upload an MP4 video file", type=["mp4"])
    youtube_url = st.sidebar.text_input("Or Paste YouTube URL")

    # Initialize a variable to determine if the user is ready to generate a report
    ready_to_generate = False

    if token and (uploaded_file is not None or youtube_url != ""):
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.success("Token verified.")
        
        if uploaded_file is not None:
            st.write("Filename:", uploaded_file.name)
        elif youtube_url != "":
            st.write("YouTube URL:", youtube_url)
        
        ready_to_generate = True

    else:
        st.sidebar.warning("Please enter the Claude API Key to generate the report.")
    
    # Render the "Generate Report" button if ready
    if ready_to_generate and st.sidebar.button("Generate Report"):
        st.write("Sit back, it's gonna take a minute or two.")
        
        if uploaded_file is not None:
            # Save uploaded file to disk and process
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            generate_report(f"C:/Users/orile/Documents/TutorialDigest/{uploaded_file.name}", selected_model,client)
            os.remove(uploaded_file.name)  # Delete the uploaded file from disk after processing
            st.success("Report generated successfully!")
            
        elif youtube_url != "":
            # Process the YouTube video
            path = download_youtube_video(youtube_url, "C:/Users/orile/Documents/TutorialDigest/")
            generate_report(path, selected_model,client)
            st.success("Report generated successfully!")




if __name__ == "__main__":
    main()

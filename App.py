import re
import shutil
import subprocess
import os
from pydub import AudioSegment
import speech_recognition as sr
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize
import random
from PIL import Image
import google.generativeai as genai
import base64
from pytube import YouTube

def process_video(input_video_path, Gemini_Api_Key):
    genai.configure(api_key=Gemini_Api_Key)

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
    def extract_audio_ffmpeg(input_file, output_file):
        print("Extracting audio...", end='\r')
        # Construct the FFmpeg command
        command = ['ffmpeg', '-i', input_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', output_file]
        # Execute the command
        subprocess.run(command, check=True)
        print("Extracting audio complete")

    # Function to transcribe audio using SpeechRecognition
    def transcribe_audio(input_wav,csv_file_path, interval=15000):
        print("Transcription process started")
        audio = AudioSegment.from_wav(input_wav)
        segments = [audio[i:i+interval] for i in range(0, len(audio), interval)]
        recognizer = sr.Recognizer()

        with open(csv_file_path, 'w', newline='') as csvfile:
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
                img_path = selected_screenshot
                img = Image.open(img_path)
                model = genai.GenerativeModel('gemini-pro-vision')
                response = model.generate_content([text, img], stream=True)
                response.resolve()
                generated_response = response.text  # Placeholder for actual response

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
    
    def urlOrFile(input_video_path):
        # Check if the input is a YouTube URL
        if input_video_path.startswith('https://www.youtube.com/') or input_video_path.startswith('http://www.youtube.com/') or input_video_path.startswith('www.youtube.com/'):
            input_video_path = download_youtube_video(input_video_path)
        return input_video_path
    
    # Create a subfolder with the basename of the input video path
    video_folder_name = os.path.basename(input_video_path).split('.')[0]
    #video_folder_path = os.path.join(output_folder, video_folder_name)
    #os.makedirs(video_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Define paths and filenames
    base_name = os.path.basename(video_folder_name)
    audio_output = os.path.join(video_folder_name, f'{base_name[:-4]}_audio.wav')
    screenshot_folder = os.path.join(video_folder_name, f'{base_name[:-4]}_Screenshots')
    os.makedirs(screenshot_folder, exist_ok=True)  # Create the screenshots folder
    csv_file_path = os.path.join(video_folder_name, 'transcriptions.csv')
    output_csv_file_path = os.path.join(video_folder_name, 'updated_transcriptions.csv')
    html_file_path = os.path.join(video_folder_name, 'output.html')

    # Step 0: YouTube Url or Actual file
    input_video_path = urlOrFile(input_video_path)

    # Step 1: Extract Audio
    extract_audio_ffmpeg(input_video_path, audio_output)

    # Step 2: Transcribe Audio
    transcribe_audio(audio_output,csv_file_path)

    # Step 3: Extract Screenshots
    extract_screenshots_ffmpeg(input_video_path, screenshot_folder)

    # Step 4: Delete Similar Following Frames - No Need
    # delete_similar_following_frames(screenshot_folder)

    # Step 5: Select Random Screenshot and Update CSV
    select_random_screenshot_and_update_csv(screenshot_folder, csv_file_path, output_csv_file_path)

    # Step 6: Convert CSV to HTML
    csv_to_html(output_csv_file_path, html_file_path, screenshot_folder)

    # Remove the subfolder after processing
    #print("Cleaning up...")
    #shutil.rmtree(video_folder_name)
    print("-----------------------")
    print(f"Open up the HTML in your browser to see the report\nDont forget to do Cleanup to the {video_folder_name} when you done.")
    print("-----------------------")
    
# Input -> YouTube URL or directory of a file, Output -> HTML file, just open it with browser
input_video_path = 'The_Perfect_Snowball.mp4'
Gemini_Api_Key = 'GoogSyDim1J1f_zqlkeyKeyukcgXCn0o62Tqoaw'
process_video(input_video_path, Gemini_Api_Key)

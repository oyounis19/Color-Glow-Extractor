import os
import streamlit as st
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ColorBorderProcessor:
    @staticmethod
    def extract_dominant_colors(input_data, num_clusters=5, color_tolerance=10):
        """
        Extract and sort dominant colors.
        Works for both images and video frames.
        
        Args:
            input_data (np.ndarray): Input image or video frame
            num_clusters (int): Number of color clusters
        
        Returns:
            tuple: (sorted colors, sorted percentages, clustered image)
        """
        # Reshape input to 2D array of pixels
        pixels = input_data.reshape((-1, 3))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(pixels)
        
        # Calculate color percentages
        labels = kmeans.labels_
        color_percentages = np.bincount(labels) / len(labels) * 100
        
        # Extract dominant colors
        dominant_colors = kmeans.cluster_centers_.astype(int)

        # Clip color values to 0-255 range
        dominant_colors = np.clip(dominant_colors, 0, 255)
        
        # Reconstruction of clustered image
        clustered_image = dominant_colors[kmeans.labels_].reshape(input_data.shape)

        # Compute colorfulness for each cluster
        colorfulness = [ColorBorderProcessor.compute_colorfulness(color) for color in dominant_colors]

        if np.max(colorfulness) < color_tolerance:
            # If the most colorful cluster is below the threshold, set to gray (set first index to gray)
            dominant_colors = np.insert(dominant_colors, 0, [230, 230, 230], axis=0)
            color_percentages = np.insert(color_percentages, 0, 100)
        else:
            # Sort colors by colorfulness
            dominant_colors, color_percentages, colorfulness = zip(*sorted(
                zip(dominant_colors, color_percentages, colorfulness),
                key=lambda x: x[2], reverse=True
            ))
        return dominant_colors, color_percentages, clustered_image, colorfulness

    @staticmethod
    def compute_colorfulness(color):
        """
        Compute the colorfulness of a given RGB color.
        
        Implementation based on Hasler and Süsstrunk's colorfulness metric.
        """
        r, g, b = color
        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)

        # Compute the mean and standard deviation of both `rg` and `yb`.
        rg_mean, rg_std = (np.mean(rg), np.std(rg))
        yb_mean, yb_std = (np.mean(yb), np.std(yb))

        # Combine the mean and standard deviations.
        std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))

        return std_root + (0.3 * mean_root)
    
    @staticmethod
    def add_color_border(image, border_color, border_width=50):
        """
        Add a color border to an image.
        
        Args:
            image (np.ndarray): Input image
            border_color (tuple): RGB border color
            border_width (int): Width of the border
        
        Returns:
            np.ndarray: Image with colored border
        """
        height, width = image.shape[:2]
        bordered_image = np.full(
            (height + 2*border_width, width + 2*border_width, 3), 
            border_color, 
            dtype=np.uint8
        )
        
        # Place original image in the center
        bordered_image[
            border_width:border_width+height, 
            border_width:border_width+width
        ] = image
        
        return bordered_image

class StreamlitColorBorderApp:
    def __init__(self):
        """
        Initialize Streamlit app with title and description
        """
        st.title('Dynamic Color Border Analyzer')
        st.write("""
        Upload an image or video to automatically generate a dynamic color border 
        based on the dominant colors in the content.
        """)
        
        # File uploader
        self.uploaded_file = st.file_uploader(
            "Choose an image or video", 
            type=['jpg', 'jpeg', 'png', 'mp4', 'avi']
        )
        
        # Processing options
        self.num_colors = st.slider(
            'Number of dominant colors to extract (Higher values may take longer)', 
            min_value=1, 
            max_value=10, 
            value=5
        )
        
        # Processing button
        self.process_button = st.button('Process and Visualize')

    def process_image(self, image_array):
        """
        Process a single image and show results
        """
        # Convert to RGB if needed
        if image_array.shape[-1] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Extract dominant colors
        dominant_colors, percentages, clustered_image, colorfulness = ColorBorderProcessor.extract_dominant_colors(
            image_array, 
            num_clusters=self.num_colors
        )
        
        # Add color border with most dominant color
        most_dominant_color = dominant_colors[0]
        bordered_image = ColorBorderProcessor.add_color_border(
            image_array, 
            most_dominant_color
        )
        
        # Create visualization columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Original Image')
            st.image(image_array, use_container_width=True)
        
        with col2:
            st.subheader('Image with Color Border')
            st.image(bordered_image, use_container_width=True)
        
        # Display color analysis
        st.subheader('Color Analysis')

        # Check if there is no colorfulness in the image
        if len(dominant_colors) > self.num_colors:
            dominant_colors = dominant_colors[1:]
            percentages = percentages[1:]

        fig_percentages, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            percentages, 
            colors=[color/255 for color in dominant_colors], 
            autopct='%1.1f%%', 
            startangle=90
        )
        
        plt.setp(autotexts, size=10, weight="bold")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Clustered Image')
            st.image(clustered_image, use_container_width=True)
        with col2:
            st.subheader('Color Percentages')
            st.pyplot(fig_percentages)

        # Show dominant colors
        st.subheader('Dominant Colors')
        color_display = np.zeros((100, len(dominant_colors)*100, 3), dtype=np.uint8)
        for i, color in enumerate(dominant_colors):
            color_display[:, i*100:(i+1)*100] = color
        
        st.image(color_display, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Show colorfulness values
            st.subheader('Colorfulness Values:')
            st.write("Colorfulness is calculated based on the Hasler and Süsstrunk's colorfulness metric. "
                     "It measures the standard deviation of the color channels and the mean color difference. "
                     "Higher values indicate more colorful clusters. "
                     "If the most colorful cluster is below the threshold (10), the border color is set to gray. "
                     "For more details, refer to their [research paper](https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images).")
        with col2:
            # Show colorfulness bar chart
            fig_colorfulness, ax = plt.subplots(figsize=(3, 3))
            ax.bar(range(len(colorfulness)), colorfulness, color=[color/255 for color in dominant_colors])
            ax.set_xticks(range(len(colorfulness)))
            ax.set_ylabel('Colorfulness')
            st.pyplot(fig_colorfulness)

    def process_video(self, video_path):
        """
        Process video with dynamic color border
        """
        # Collect dominant colors for each frame group
        cap = cv2.VideoCapture(video_path)
        frame_colors = []
        frame_count = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get dominant colors for this frame
            dominant_colors, _, _, _ = ColorBorderProcessor.extract_dominant_colors(
                frame_rgb,
                num_clusters=self.num_colors
            )
            frame_colors.append(dominant_colors[0])

            # Skip the next 9 frames
            frame_count += 10
        
        cap.release()
        
        # Play video with corresponding border colors
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1/fps  # Calculate delay between frames

        st.subheader('Video with Dynamic Color Border')
        video_placeholder = st.empty()
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get corresponding border color
            color_idx = min(frame_idx // 10, len(frame_colors)-1)
            if color_idx == len(frame_colors)-1:
                print('End of video reached.')

            border_color = frame_colors[color_idx]
            
            # Add color border
            bordered_frame = ColorBorderProcessor.add_color_border(
                frame_rgb, 
                border_color
            )
            
            diff = time.time() - current_time
            if diff  < frame_delay:
                time.sleep(frame_delay - diff)
            
            # Display bordered frame
            video_placeholder.image(bordered_frame, channels='RGB', use_container_width=True)
            frame_idx += 1
        
        cap.release()
        
        # Show dominant colors
        st.subheader('Video Dominant Colors')
        color_display = np.zeros((100, len(dominant_colors)*100, 3), dtype=np.uint8)
        for i, color in enumerate(dominant_colors):
            color_display[:, i*100:(i+1)*100] = color
        
        st.image(color_display, use_container_width=True)

    def run(self):
        """
        Main application runner
        """
        if self.uploaded_file is not None:
            # Determine file type
            file_extension = self.uploaded_file.name.split('.')[-1].lower()
            
            # Read file
            file_bytes = np.asarray(bytearray(self.uploaded_file.read()), dtype=np.uint8)
            
            if file_extension in ['jpg', 'jpeg', 'png']:
                # Image processing
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if self.process_button:
                    self.process_image(image)
            
            elif file_extension in ['mp4', 'avi']:
                # Video processing
                # Write uploaded file to temporary location
                with open(f'temp_video.{file_extension}', 'wb') as f:
                    f.write(file_bytes)
                
                if self.process_button:
                    self.process_video(f'temp_video.{file_extension}')

                # Remove temporary video file
                os.remove(f'temp_video.{file_extension}') if os.path.exists(f'temp_video.{file_extension}') else None
            else:
                st.write('Invalid file format. Please upload an image or video.')

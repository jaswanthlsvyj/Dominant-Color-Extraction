# Dominant Color Extraction App

This is a Streamlit web application that allows users to upload an image and extract the dominant colors from it. The app also displays a recolored version of the image using the extracted dominant colors.

## Features

- Upload an image (jpg, jpeg, or png).
- Select the number of dominant colors to extract.
- View the original image and the recolored image with dominant colors.
- Display the top dominant colors in the image.

**Website:** [dominant-color-extraction-app](https://dominant-color-extraction-app.streamlit.app/)

## How to Use

1. **Navigate to the Upload Image Page**:
   - Use the sidebar to select "Upload Image".

2. **Upload an Image**:
   - Click on the "Choose an image..." button and upload your image file.

3. **Select the Number of Dominant Colors**:
   - Use the slider to choose the number of dominant colors to extract.

4. **View the Results**:
   - The original image and the recolored image will be displayed side by side.
   - The top dominant colors will be shown below the images.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/jaswanthlsvyj/Dominant-Color-Extraction.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Dominant-Color-Extraction
    ```

3. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

5. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## Dependencies

- streamlit
- pillow
- numpy
- scikit-learn

## Project Structure

- `app.py`: Main application file.
- `requirements.txt`: List of required Python packages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for providing the web framework.
- [Pillow](https://python-pillow.org/) for image processing.
- [scikit-learn](https://scikit-learn.org/) for the KMeans clustering algorithm.



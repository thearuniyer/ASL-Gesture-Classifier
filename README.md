# ASL Gesture Prediction :v:

This project focuses on predicting American Sign Language (ASL) gestures using machine learning techniques. The goal is to provide a tool that can interpret ASL gestures and help improve communication for the hearing-impaired.

## Project Files :file_folder:

The following files are included in this project:

- `keypoints.json`: Extracts the keypoints of body parts from the frames of the videos.
- `convert_to_csv.py`: Converts the json file of keypoints to its csv equivalent.
- `asl_feat_preprocessing_final.py`: Calculates the positions of the body parts w.r.t nose and removes the unnecessary attributes for further analysis.
- `asl_feat_extracxn_final.py`: Implements feature extraction techniques to obtain the 1st, 2nd, 3rd degree FFT & PSD coefficients, autocorrelation coefficients with lags, and other statistical properties.
- `asl_model_final.py`: Trains, validates and tests ML models on the dataset.
- `data_X.txt`: The dataset obtained after analysis and data wrangling.
- `data_Y.txt`: The labels for `data_X`.
- `app.py`: Server program that delivers predicted results for the passed json file on POST. The `app.py` program was tested using POSTMAN as the client.

## ASL Gestures :hand:

This project focuses on six ASL gestures: 

- Buy
- Communicate
- Fun
- Hope
- Mother
- Really

## Models :brain:

The models have been pickled and can be found in the `models` folder.

## Contributing :rocket:

Contributions are welcome! If you find any bugs or have suggestions for improvement, please feel free to open an issue or a pull request.

## License :page_with_curl:

This project is licensed under the MIT License - see the LICENSE.md file for details.
The server url is : http://18.223.185.34:80/

# **F1 Race Outcome Prediction – Random Forest (Python, scikit-learn, ML)**

This project builds a machine learning pipeline that predicts Formula 1 race finishing positions using driver, constructor, grid, and lap-time data. Using a Random Forest Regressor with hyperparameter optimization, it models historical race results to forecast finishing position with high ±1 position prediction accuracy.

## **Overview**

* Loads and merges ~50,000 historical F1 records (drivers, constructors, race results, standings).
* Cleans and normalizes the dataset (handling missing values, converting fastest lap times into numeric values, processing DNFs).
* Engineers predictive features such as driver/constructor rankings, grid position, laps completed, and fastest lap performance.
* Encodes categorical driver/constructor identifiers with one-hot encoding.
* Trains a Random Forest Regression model, optimized with RandomizedSearchCV using a custom ±1 position accuracy metric.
* Outputs prediction accuracy metrics including RMSE, R², and ±1 position accuracy.

## **Features**

* **End-to-end automation:** Cleans, preprocesses, encodes, trains, evaluates, and outputs metrics.
* **Feature Engineering:** Converts lap time strings, assigns consistent values for DNFs, extracts ranking & points stats.
* **Random Forest Modeling:** Uses RandomizedSearchCV to optimize model hyperparameters.
* **Custom Accuracy Score:** Evaluates performance based on ±1 finishing position accuracy.
* **Scalable Data Handling:** Processes large historical F1 datasets across seasons and driver/constructor lineups.

## **How It Works**

1.  **Data Loading:**
    * Reads all CSV files from the archive/ directory (results, races, drivers, constructors, standings).
2.  **Data Cleaning & Preprocessing**
    * Converts fastest lap time (e.g., "1:22.325") → numeric seconds.
    * Handles DNF (Did Not Finish) entries by assigning:
      * Worst possible finishing position + 1
      * Slowest-fastest-lap + 1 second
      * Lowest ranking / 0 points
3.  **Feature Engineering**
    * Selects predictive fields:
      * driver & constructor standings, grid, laps, fastest lap time, rankings, and points.
    * One-hot encodes driver ID and constructor ID → ML-ready numerical matrix.
4.  **Model Training**
    * Splits into training/testing sets (train_test_split).
    * Runs RandomizedSearchCV over:
      * number of trees
      * depth
      * feature selection strategy
      * leaf size
      * etc.
    * Uses custom scorer:
      * abs(actual - predicted) <= 1
5.  **Evaluation**
    * Prints:
      * Best model hyperparameters
      * RMSE
      * R² score
      * ±1 position prediction accuracy


## **Setup and Installation**

### **Recommended: Google Colaboratory**

This project is ideally suited for Google Colab, as it provides a free environment with pre-installed libraries and sufficient computing resources for most runs.

1.  **Upload Shapefile:** Upload the US county shapefile (`cb_2018_us_county_500k.shp`) to your Colab session (e.g., into the `/content/sample_data/` folder, or adjust the path in the code).
2.  **Run Cells:** Simply paste the provided Python code into a Colab notebook cell and run it.

### **Local Environment**

If you prefer to run the project locally, ensure you have Python 3.7+ installed along with the following libraries. It's often recommended to use a virtual environment.

1.  **Download Shapefile:** Download the `cb_2018_us_county_500k.shp` file (and its accompanying `.dbf`, `.shx`, `.prj`, etc., files) from the [US Census Bureau website](https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip) and place them in your project directory.
2.  **Install Dependencies:**
    ```bash
    pip install pandas geopandas matplotlib mapclassify imageio joblib numpy
    ```
    *(Note: `geopandas` can sometimes be complex to install locally due to its geospatial dependencies. Using `conda` or referring to the [Geopandas installation guide](https://geopandas.org/en/stable/getting_started/install.html) might be helpful if you encounter issues.)*
3.  **Run Script:** Execute your Python script from your terminal:
    ```bash
    python your_script_name.py
    ```

## **Usage**

1.  **Execute the Code:** Run all the cells in your Google Colab notebook or execute the Python script.
2.  **Monitor Progress:** The script will print messages to the console indicating which frame it's processing.
3.  **View Output:**
    * Upon completion, the animated GIF (`covid_animation.gif`) will be displayed directly within your notebook's output cell.
    * The GIF file will also be saved to the main directory of your Colab session (or your script's directory if running locally), allowing you to download or share it.

## **Customization Options**

You can easily modify the animation's behavior and appearance by adjusting parameters in the code:

* **Frame Interval (`dates_to_process`):**
    * Located in the `Merging Data Set With US Map` section.
    * Change `dates_to_process = all_date_columns[::30]` to:
        * `::1` for every single day (slowest, most frames).
        * `::7` for every week.
        * `::14` for every two weeks.
        * A larger number results in fewer frames and a faster overall process.
* **Animation Speed (FPS):**
    * Located in the `Animation Output` section.
    * Modify `fps=1` in `imageio.get_writer(...)`.
    * `fps=10` makes the animation play 10 frames per second (faster).
    * `fps=1` makes it play 1 frame per second (slower, default for this code).
* **Image Quality/Resolution (DPI):**
    * Located within the `generate_single_frame` function.
    * Adjust `dpi=50` in `plt.savefig(..., dpi=50, ...)`.
    * Lower DPI (e.g., `30`) means faster processing and smaller GIF size, but lower image quality.
    * Higher DPI (e.g., `100`) means better quality but slower processing and larger GIF size.
* **Map Size (`figsize`):**
    * Located within the `generate_single_frame` function.
    * Modify `figsize=(12, 7)` in `plt.subplots(...)` to change the width and height of each map image.
* **Number of Parallel Workers (`n_jobs`):**
    * Located in the `Merging Data Set With US Map` section (in the `Parallel` call).
    * `n_jobs=-1` (default) uses all available CPU cores, recommended for maximum speed.
    * You can set a specific number (e.g., `n_jobs=4`) if you want to limit worker count.
* **Background Color:**
    * Located within the `generate_single_frame` function.
    * The current code sets the background to black (`facecolor='black'` and `ax.set_facecolor('black')`) and the title color to white (`color='white'`). You can modify these values to your preferred colors.

## **Data Source**

* **Formula 1 World Championship (1950–2020):**
    https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?utm_source=chatgpt.com


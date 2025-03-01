import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
import joblib
from __init__ import MLWorkflow

class MLApp:
    """
        A GUI application for managing a Machine Learning pipeline including:
        - Loading train and test datasets
        - Selecting target and feature columns
        - Configuring model parameters for Gradient Boosting, Decision Tree, and Random Forest
        - Training, testing, and making predictions using trained models

        Utilizes the `MLWorkflow` class for backend ML operations.
        """

    def __init__(self, root):
        """Initializes the GUI with various sections like data loading, model configuration, training, testing, and predictions."""

        self.root = root
        self.root.title("Machine Learning Pipeline Interface")

        # Create scrollable frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self.content_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # -------------------- Train Data Section --------------------
        self.train_file_path = tk.StringVar()
        self.test_file_path = tk.StringVar()
        self.selected_columns = []  # Store columns from train data

        # Train File Selection
        lbl_train = tk.Label(self.content_frame, text="Select Train Dataset:")
        lbl_train.pack(pady=5)

        frame_train = tk.Frame(self.content_frame)
        frame_train.pack(pady=5)
        btn_browse_train = tk.Button(frame_train, text="Browse", command=lambda: self.load_file(is_train=True))
        btn_browse_train.pack(side=tk.LEFT)
        self.entry_train = tk.Entry(frame_train, textvariable=self.train_file_path, width=50)
        self.entry_train.pack(side=tk.LEFT, padx=5)

        # Target & Column Selection
        self.label_target = tk.Label(self.content_frame, text="Choose Target Column:")
        self.label_target.pack(pady=5)
        self.target_col = tk.StringVar()
        self.combo_target = ttk.Combobox(self.content_frame, textvariable=self.target_col)
        self.combo_target.pack(pady=5)

        self.label_columns = tk.Label(self.content_frame, text="Select columns to keep:")
        self.label_columns.pack(pady=5)
        self.columns_listbox = tk.Listbox(self.content_frame, selectmode=tk.MULTIPLE, height=6)
        self.columns_listbox.pack(pady=5)

        self.btn_submit_train = tk.Button(self.content_frame, text="Process Train Data",
                                          command=self.process_train_data)
        self.btn_submit_train.pack(pady=5)

        # -------------------- Model Selection & Parameters --------------------
        self.label_model = tk.Label(self.content_frame, text="Choose Model:")
        self.label_model.pack(pady=5)
        self.model_var = tk.StringVar()
        self.combo_model = ttk.Combobox(self.content_frame, textvariable=self.model_var,
                                        values=["Gradient Boosting", "Decision Tree", "Random Forest"])
        self.combo_model.pack(pady=5)
        self.combo_model.current(0)

        self.btn_submit_model = tk.Button(self.content_frame, text="Submit Model", command=self.submit_model)
        self.btn_submit_model.pack(pady=5)

        self.label_params = tk.Label(self.content_frame, text="Model Parameters:")
        self.label_params.pack(pady=5)
        self.param_frame = tk.Frame(self.content_frame)
        self.param_frame.pack(pady=5)

        # Train button section
        self.btn_train = tk.Button(self.content_frame, text="Train Model", command=self.train_model)
        self.btn_train.pack(pady=5)

        # -------------------- Test Data & Model Testing --------------------
        lbl_test = tk.Label(self.content_frame, text="Select Test Dataset:")
        lbl_test.pack(pady=5)

        frame_test = tk.Frame(self.content_frame)
        frame_test.pack(pady=5)
        btn_browse_test = tk.Button(frame_test, text="Browse", command=lambda: self.load_file(is_train=False))
        btn_browse_test.pack(side=tk.LEFT)
        self.entry_test = tk.Entry(frame_test, textvariable=self.test_file_path, width=50)
        self.entry_test.pack(side=tk.LEFT, padx=5)

        self.btn_submit_test = tk.Button(self.content_frame, text="Process Test Data",
                                         command=self.process_test_data)
        self.btn_submit_test.pack(pady=5)

        self.btn_test = tk.Button(self.content_frame, text="Test Model", command=self.test_model)
        self.btn_test.pack(pady=5)

        self.label_accuracy = tk.Label(self.content_frame, text="Accuracy: N/A")
        self.label_accuracy.pack(pady=5)

        # -------------------- Custom Prediction Section --------------------
        # This section automatically creates one input field for each column
        # (from the "Select columns" list excluding the target) after train data is processed.
        separator = ttk.Separator(self.content_frame, orient='horizontal')
        separator.pack(fill='x', pady=10)

        self.label_custom = tk.Label(self.content_frame, text="Diabetes diagnosis of individual")
        self.label_accuracy.pack(pady=5)
        # self.label_custom = tk.Label(self.content_frame, text="On binary questions: yes: 1, no: 0")
        # self.label_accuracy.pack(pady=5)
        # self.label_custom = tk.Label(self.content_frame, text="Smoking history:")
        # self.label_accuracy.pack(pady=5)
        # self.label_custom = tk.Label(self.content_frame, text="No Info: 0, never: 1, current: 2, former: 3, ever: 4, current: 5")
        # self.label_custom.pack(pady=5)

        self.custom_frame = tk.Frame(self.content_frame)
        self.custom_frame.pack(pady=5)

        self.btn_predict_custom = tk.Button(self.content_frame, text="Predict Custom Data Point", command=self.predict_custom_input)
        self.btn_predict_custom.pack(pady=5)

    # -------------------- File Loading --------------------
    def load_file(self, is_train=True):
        """Loads a CSV file, updates target and feature selections if it's a train dataset."""

        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            if is_train:
                self.train_file_path.set(file_path)
                try:
                    df = pd.read_csv(file_path, index_col=0)
                    self.combo_target['values'] = list(df.columns)
                    self.columns_listbox.delete(0, tk.END)
                    for col in df.columns:
                        self.columns_listbox.insert(tk.END, col)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load file: {e}")
            else:
                self.test_file_path.set(file_path)

    # -------------------- Model Parameters --------------------
    def submit_model(self):
        """Displays and sets up input fields for model parameters based on the selected model."""

        for widget in self.param_frame.winfo_children():
            widget.destroy()

        model_name = self.model_var.get()
        self.params = {}
        params_dict = {
            "Gradient Boosting": {
                "n_estimators": ("Number of boosting stages (default: 100)", "100"),
                "learning_rate": ("Shrinks the contribution of each tree (default: 0.1)", "0.1"),
                "max_depth": ("Maximum depth of individual trees (default: 3)", "3"),
                "min_samples_split": ("Minimum samples required to split a node (default: 2)", "2"),
                "min_samples_leaf": ("Minimum samples required at a leaf node (default: 1)", "1"),
                "subsample": ("Fraction of samples used for fitting (default: 1.0)", "1.0"),
                "max_features": ("Number of features to consider for splits (default: None)", "None"),
                "min_impurity_decrease": ("Minimum impurity decrease for split (default: 0.0)", "0.0"),
            },
            "Decision Tree": {
                "max_depth": ("Maximum depth of the tree (default: None)", "None"),
                "min_samples_split": ("Minimum samples required to split a node (default: 2)", "2"),
                "min_samples_leaf": ("Minimum samples required at a leaf node (default: 1)", "1"),
                "max_features": ("Number of features to consider for splits (default: None)", "None"),
                "min_impurity_decrease": ("Minimum impurity decrease for split (default: 0.0)", "0.0"),
            },
            "Random Forest": {
                "n_estimators": ("Number of trees in the forest (default: 100)", "100"),
                "max_depth": ("Maximum depth of the trees (default: None)", "None"),
                "min_samples_split": ("Minimum samples required to split a node (default: 2)", "2"),
                "min_samples_leaf": ("Minimum samples required at a leaf node (default: 1)", "1"),
                "max_features": ("Number of features to consider for splits (default: 'sqrt')", "'sqrt'"),
                "min_impurity_decrease": ("Minimum impurity decrease for split (default: 0.0)", "0.0"),
            }
        }

        for param, (description, default_value) in params_dict.get(model_name, {}).items():
            label = tk.Label(self.param_frame, text=f"{param}: {description}")
            label.pack()
            entry = tk.Entry(self.param_frame)
            entry.insert(0, default_value)
            entry.pack()
            self.params[param] = entry

    # -------------------- Data Processing --------------------
    def process_train_data(self):
        """Processes and saves the training dataset with selected features and target column."""

        selected_indices = self.columns_listbox.curselection()
        self.selected_columns = [self.columns_listbox.get(i) for i in selected_indices]
        target_column = self.target_col.get()

        if not target_column:
            messagebox.showerror("Error", "Please select a target column")
            return

        if target_column not in self.selected_columns:
            self.selected_columns.append(target_column)

        try:
            df = pd.read_csv(self.train_file_path.get())
            df = df[self.selected_columns]
            df.to_csv("train_processed.csv", index=False)
            messagebox.showinfo("Success", "Train data processed successfully!")
            # Create custom input fields for each feature (exclude the target)
            self.generate_custom_inputs()
        except Exception as e:
            messagebox.showerror("Error", f"Train processing failed: {e}")

    def process_test_data(self):
        """Processes and saves the test dataset with the same columns as train data."""

        if not self.selected_columns:
            messagebox.showerror("Error", "Process train data first!")
            return

        try:
            df = pd.read_csv(self.test_file_path.get(), index_col=0)
            df = df[self.selected_columns]
            df.to_csv("test_processed.csv", index=False)
            messagebox.showinfo("Success", "Test data processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Test processing failed: {e}")

    # -------------------- Model Training --------------------
    def train_model(self):
        """Trains the selected model with specified parameters and displays the accuracy."""

        params = {}
        for key, entry in self.params.items():
            value = entry.get().strip()
            if value:
                try:
                    params[key] = int(value)
                except ValueError:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        if value.lower() == "none":
                            params[key] = None
                        elif value.lower() == "'sqrt'":
                            params[key] = "sqrt"
                        else:
                            params[key] = value

        workflow = MLWorkflow(models_dir='models')
        try:
            random_state = int(pd.Timestamp.now().timestamp() % 1000)

            acc = workflow.train_model("train_processed.csv", self.target_col.get(),
                                       model_name=self.model_var.get(),
                                       model_params=params,
                                       random_state=random_state)
            self.label_accuracy.config(text=f"Accuracy: {acc:.2f}")
            self.root.update()
            messagebox.showinfo("Training Complete", f"Model trained successfully! Accuracy: {acc:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")

    # -------------------- Model Testing --------------------
    def test_model(self):
        """Tests the trained model on the processed test dataset and displays the accuracy."""

        workflow = MLWorkflow(models_dir='models')
        try:
            # Instead of using self.test_file_path.get(), use the processed test file.
            test_file_path = "test_processed.csv"
            acc = workflow.test_model(test_file_path, self.target_col.get())
            self.label_accuracy.config(text=f"Accuracy: {acc:.2f}")
            self.root.update()
            messagebox.showinfo("Testing Complete", f"Model tested successfully! Accuracy: {acc:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {e}")

    # -------------------- Custom Input Generation & Prediction --------------------
    def generate_custom_inputs(self):
        """Generates input fields for making predictions on new data points."""

        # Clear any existing custom input fields
        for widget in self.custom_frame.winfo_children():
            widget.destroy()
        self.custom_entries = {}
        target = self.target_col.get()
        for col in self.selected_columns:
            if col == target:
                continue
            frame = tk.Frame(self.custom_frame)
            frame.pack(fill='x', pady=2)
            label = tk.Label(frame, text=col, width=20, anchor='w')
            label.pack(side='left')
            entry = tk.Entry(frame)
            entry.pack(side='left', fill='x', expand=True)
            self.custom_entries[col] = entry

    def predict_custom_input(self):
        """Predicts the target based on user-input data using the latest trained model."""

        if not hasattr(self, 'custom_entries') or not self.custom_entries:
            messagebox.showerror("Error", "Custom input fields are not available. Process train data first!")
            return
        try:
            input_data = {}
            for col, entry in self.custom_entries.items():
                value = entry.get().strip()
                if value == "":
                    messagebox.showerror("Error", f"Please provide a value for {col}.")
                    return
                try:
                    input_data[col] = float(value)
                except ValueError:
                    input_data[col] = value
            input_df = pd.DataFrame([input_data])

            # Load the latest trained model from the models directory
            workflow = MLWorkflow(models_dir='models')
            model_files = [f for f in os.listdir(workflow.models_dir) if f.endswith('.pkl')]
            if not model_files:
                messagebox.showerror("Error", "No trained models found. Please train a model first.")
                return

            # Assumes the naming convention {id}_model_{num}.pkl and uses the highest number (latest)
            model_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            latest_model_path = os.path.join(workflow.models_dir, model_files[-1])
            model = joblib.load(latest_model_path)

            # If the model stores the feature names, reorder the input DataFrame accordingly
            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
                missing_cols = set(expected_cols) - set(input_df.columns)
                if missing_cols:
                    messagebox.showerror("Error", f"Missing columns for prediction: {missing_cols}")
                    return
                input_df = input_df[expected_cols]

            prediction = model.predict(input_df)[0]
            messagebox.showinfo("Prediction Result", f"Predicted target: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
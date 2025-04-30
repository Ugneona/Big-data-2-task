from flask import Flask, Response
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix

app = Flask(__name__)

@app.route("/")
def index():
    # Load and prepare the Iris dataset
    df = px.data.iris()  # Use plotly's built-in Iris dataset
    X = df.drop(columns=["species", "species_id"])  # Features
    y = df["species"]  # Target variable (species)
    
    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)  # Fit the model on the training data
    y_pred = model.predict(X_test)  # Predict labels for the test data
    y_scores = model.predict_proba(X_test)  # Get probability estimates for the test data

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)  # Generate confusion matrix from predictions

    # Convert confusion matrix into a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=y.unique(), columns=y.unique())

    # Create a heatmap visualization of the confusion matrix
    fig_cm = px.imshow(cm_df, 
                       text_auto=True,  # Automatically add the numeric values in the heatmap
                       color_continuous_scale='Viridis',  # Color scale for the heatmap
                       title="Confusion Matrix Heatmap")  # Title for the heatmap

    fig_cm.update_layout(
        height=700,
        width=900,
    )

    fig_cm_html = fig_cm.to_html()  # Convert the plot into HTML format for embedding

    # Generate the classification report (Precision, Recall, F1-Score, Accuracy)
    report = classification_report(y_test, y_pred, digits=4)  # Compute classification report

    # Primary feature analysis (subplots of the Iris dataset's features)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    fig_primary = make_subplots(rows=2, cols=2, subplot_titles=features)  # Create subplots grid
    row_col = [(1, 1), (1, 2), (2, 1), (2, 2)]  # Define subplot positions

    # Track species to ensure one legend entry per species
    shown_species = set()

    # Loop over each feature and add line plots to subplots
    for i, feature in enumerate(features):
        fig = px.line(df, y=feature, color='species')  # Create line plot for each feature
        for trace in fig.data:
            # Only show legend for a species the first time it appears
            show_legend = trace.name not in shown_species
            if show_legend:
                shown_species.add(trace.name)  # Add species to the shown_species set
            trace.showlegend = show_legend  # Set legend visibility for each trace
            fig_primary.add_trace(trace, row=row_col[i][0], col=row_col[i][1])  # Add plot to the subplot grid

    # Update layout for the primary feature analysis plot
    fig_primary.update_layout(
        title_text="Primary Feature Analysis (Iris Dataset)",
        height=700,
        width=900,
        legend_title="Species"  # Title for the legend
    )

    fig_primary_html = fig_primary.to_html()  # Convert subplot figure to HTML format

    # Generate ROC curves for each class (multiclass classification)
    y_onehot = pd.get_dummies(y_test, columns=model.classes_)  # One-hot encode the true labels
    fig_roc = go.Figure()
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)  # Diagonal line (random classifier)

    # Loop over each class and plot the ROC curve
    for i in range(y_scores.shape[1]):
        fpr, tpr, _ = roc_curve(y_onehot.iloc[:, i], y_scores[:, i])  # Compute ROC curve
        auc_score = roc_auc_score(y_onehot.iloc[:, i], y_scores[:, i])  # Compute AUC score
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{y_onehot.columns[i]} (AUC={auc_score:.2f})", mode='lines'))  # Add ROC curve trace

    # Update layout for the ROC curve plot
    fig_roc.update_layout(
        title="ROC Curves (Test Set)", 
        xaxis_title="False Positive Rate", 
        yaxis_title="True Positive Rate", 
        height=500, 
        width=900
    )

    fig_roc_html = fig_roc.to_html(full_html=False, include_plotlyjs=False)  # Convert ROC plot to HTML format

    # Combine all the HTML content for the response
    html = f"""
    <h2>Primary Analysis Plots</h2>
    {fig_primary_html}
    <h2>Implementing Multivariate Logistic Regression - ROC Curves for Iris Test Sample</h2>
    {fig_roc_html}
    <h2>Classification Report</h2>
    <pre>{report}</pre>
    <h2>Confusion Matrix Heatmap</h2>
    {fig_cm_html}
    """

    return Response(html, mimetype='text/html')  # Return the full HTML response with all plots and reports

if __name__ == "__main__":
       app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)



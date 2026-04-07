import streamlit as st
from streamlit_lottie import st_lottie
import json
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def set_page_configuration():
    st.set_page_config(page_title = 'PRISM (Particle Collider Event Classifier)',
                       page_icon = '⚛️',
                       layout = 'wide',
                       initial_sidebar_state = 'expanded')

def get_selected_navbar_menu():
    selected_navbar_menu = option_menu(
        menu_title = None, # required
        options = ['Main Menu', 'Prediction'],
        icons = ['house', 'search'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'horizontal',
        styles = {'nav-link': {'--hover-color': "#005eb1",
                               '--active-background-color': '#eee'},
        'nav-link-selected': {'background-color': "#0088ff"},
        })

    return selected_navbar_menu

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
@st.cache_resource
def load_model():
    model = joblib.load("catboost.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def main():
    selected_navbar_menu = get_selected_navbar_menu()

    if selected_navbar_menu == "Main Menu":
        st.markdown("<h1 style='text-align: center; color: white;'>Particle Collider Event Classifier</h1>", unsafe_allow_html=True)

        lottie_coding = load_lottiefile("assets/physics icon.json")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(
                lottie_coding,
                speed=1,
                reverse=False,
                loop=True,
                quality="low",
                height=400,
                width=400,
                key=None
            )

        project_description = """
        Accurately separating rare signal (s) events from abundant background (b) is central to particle physics.
        This dataset provides reconstructed kinematic features per event, along with a per-event Weight to reflect experimental importance. The task is binary classification: predict whether an event is signal (s) or background (b).
        """
        st.write("### Project Overview")
        st.write(project_description)
        st.divider()

        features_description = """
        Identifiers & label
        EventId (unique), Target (s/b), Weight (importance)

        Derived physics

        `DER_mass_MMC`, `DER_mass_transverse_met_lep`, `DER_mass_vis`, `DER_pt_h`,
        `DER_deltaeta_jet_jet`, `DER_mass_jet_jet`, `DER_lep_eta_centrality`

        Primary objects

        Tau: `PRI_tau_pt`, `PRI_tau_eta`, `PRI_tau_phi`
        Lepton: `PRI_lep_pt`, `PRI_lep_eta`, `PRI_lep_phi`

        MET

        `PRI_met`, `PRI_met_phi`, `PRI_met_sumet`

        Jets

        `PRI_jet_num`, `PRI_jet_leading_pt`/`PRI_jet_leading_eta`/`PRI_jet_leading_phi`, `PRI_jet_subleading_pt`/`PRI_jet_subleading_eta`/`PRI_jet_subleading_phi`, `PRI_jet_all_pt`
        """
        st.write("### Features")
        st.write(features_description)
        
    if selected_navbar_menu == "Prediction":
        st.title("Event Classification")
        st.write('Upload your event data (CSV) to predict Signal (1) or Background (0).')
        
        uploaded_file = st.file_uploader('Upload Event Data (CSV)')
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded Data")
            st.dataframe(input_df.head())

            if st.button('Predict'):
                with st.spinner('Calculating physics...'):
                    model, scaler = load_model()
                    input_ids = input_df['EventId'].tolist() if 'EventId' in input_df.columns else list(range(len(input_df)))
                    
                    y_true = None
                    if 'Label' in input_df.columns:
                        y_true = input_df['Label'].map({'b': 0, 's': 1})
                        y_true_string = input_df['Label'].map({'b': 'Background (b)', 's': 'Signal (s)'})

                    proc_df = input_df.replace(-999, 0)

                    expected_scaler_cols = scaler.feature_names_in_
                    X_for_scaler = proc_df.reindex(columns=expected_scaler_cols, fill_value=0)
                    X_scaled_array = scaler.transform(X_for_scaler)
                    
                    X_scaled_df = pd.DataFrame(X_scaled_array, columns=expected_scaler_cols)
                    cols_to_drop_from_model = [
                        'PRI_jet_leading_phi', 'PRI_jet_subleading_phi',
                        'PRI_met_sumet', 'PRI_jet_all_pt', 'DER_sum_pt', 
                        'PRI_jet_leading_pt', 'DER_pt_tot', 'DER_deltaeta_jet_jet'
                    ]
                    X_final = X_scaled_df.drop(columns=cols_to_drop_from_model, errors='ignore')
                    predictions = model.predict(X_final)

                    result_df = pd.DataFrame({
                        'EventId': input_ids,
                        'Prediction': pd.Series(predictions).map({1: 'Signal (s)', 0: 'Background (b)'})
                    })

                    if y_true is not None:
                        result_df['Ground Truth'] = y_true_string.values
                    
                    st.success("Analysis Complete!")
                    st.dataframe(result_df)

                    if y_true is not None:
                        st.write("### Performance Metrics")
                        report = classification_report(y_true, predictions)
                        st.text(report)

if __name__ == "__main__":
    main()

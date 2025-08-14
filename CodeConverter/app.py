import streamlit as st
import re
import uuid
import os
from graph import CodeConversionWorkflow

st.set_page_config(page_title="Agentic Code Converter", layout="wide")

st.title("🧑‍💻 Agentic Code Converter: DataStage to PySpark")
st.write("Upload one or more DataStage XML files to batch convert them to PySpark code using an agentic workflow.")

uploaded_files = st.file_uploader(
    "Upload DataStage XML files", type=["xml", "md"], accept_multiple_files=True
)

def get_folders(base="."):
    return [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]

existing_folders = get_folders()
selected_folder = st.selectbox(
    "Select output folder (or type a new one):",
    options=["<Create new folder>"] + existing_folders,
    index=0,
    disabled=st.session_state.get("conversion_in_progress", False)  # Disable if converting
)

if selected_folder == "<Create new folder>":
    output_folder = st.text_input(
        "Enter new output folder name:",
        value="outputs",
        disabled=st.session_state.get("conversion_in_progress", False)  # Disable if converting
    )
else:
    output_folder = selected_folder

if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    selected_idx = st.session_state.get("selected_idx", 0)

    # Initialize conversion_in_progress flag
    if "conversion_in_progress" not in st.session_state:
        st.session_state["conversion_in_progress"] = False

    # Reset conversion_in_progress if user changes file selection or uploads new files
    if "last_file_names" not in st.session_state:
        st.session_state["last_file_names"] = []

    if file_names != st.session_state["last_file_names"]:
        st.session_state["conversion_in_progress"] = False
        st.session_state["last_file_names"] = file_names.copy()

    if "last_selected_file" not in st.session_state:
        st.session_state["last_selected_file"] = selected_idx

    # Dropdown for file selection (disable if converting)
    selected_file = st.selectbox(
        "Select an input file to preview:",
        options=range(len(file_names)),
        format_func=lambda i: file_names[i],
        index=selected_idx if selected_idx < len(file_names) else 0,
        key="file_selectbox",
        disabled=st.session_state["conversion_in_progress"]
    )

    if selected_file != st.session_state["last_selected_file"]:
        st.session_state["conversion_in_progress"] = False
        st.session_state["last_selected_file"] = selected_file

    # Initialize session state for expander
    if "preview_expanded" not in st.session_state:
        st.session_state["preview_expanded"] = True  # Default to expanded

    # Handle collapse logic
    if st.session_state.get("collapse_preview", False):
        st.session_state["preview_expanded"] = False
        st.session_state["collapse_preview"] = False
        st.rerun()

    # Show selected file content in an expander
    datastage_xml = uploaded_files[selected_file].read().decode("utf-8")
    with st.expander(f"Preview: {file_names[selected_file]}", expanded=st.session_state["preview_expanded"]):
        st.code(datastage_xml, language="xml")

    # The button is below the preview pane (disable if converting)
    if st.button("Batch Convert All Files", key="batch_convert_btn", disabled=st.session_state["conversion_in_progress"]):
        st.session_state["collapse_preview"] = True
        st.session_state["run_batch_conversion"] = True
        st.session_state["conversion_in_progress"] = True  # Set immediately!
        st.rerun()

    # Proceed with batch conversion logic
    if (
        not st.session_state["preview_expanded"]
        and st.session_state.get("run_batch_conversion", False)
        and st.session_state["conversion_in_progress"]
    ):
        st.session_state["run_batch_conversion"] = False  # Reset the flag
        os.makedirs(output_folder, exist_ok=True)
        info_placeholder = st.empty()
        info_placeholder.info("Batch conversion started. Please wait...")
        workflow = CodeConversionWorkflow(max_iterations=3)
        conversion_results = []
        file_status_placeholder = st.empty()
        status_placeholder = st.empty()

        for i, file in enumerate(uploaded_files):
            file_status_placeholder.info(f"Processing file {i+1} of {len(uploaded_files)}: **{file.name}**")
            file.seek(0)
            xml_content = file.read().decode("utf-8")
            work_id = str(uuid.uuid4())
            steps = []
            for event in workflow.stream_workflow(xml_content, thread_id=work_id):
                for node_name in event:
                    if node_name in ["converter", "reflector", "revisor"]:
                        agent_label = {
                            "converter": "🛠️ Code Converter Agent (Generation)",
                            "reflector": "👁️ Reflection Agent (Reviewer)",
                            "revisor": "📝 Code Converter Agent (Revision)"
                        }[node_name]
                        status_placeholder.info(f"**{agent_label} is thinking...**")
                        steps.append(agent_label)
            results = workflow.run(xml_content, thread_id=work_id)
            pyspark_code = re.sub(r"^```python\s*|```$", "", results['generation'][-1], flags=re.MULTILINE)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file.name)[0]}_converted.py")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(pyspark_code)
            # Export log
            log_text = (
                        f"Judgement: {results['judgement']}\n"
                        f"Iterations: {results['iterations']}\n"
                        f"Final Feedback: {results['reflection'][-1] if results['reflection'] else 'No feedback provided'}\n"
                        )
            log_filename = os.path.join(output_folder, f"{os.path.splitext(file.name)[0]}_log.txt")
            with open(log_filename, "w", encoding="utf-8") as log_file:
                    log_file.write(log_text)
            
            conversion_results.append((file.name, output_path, pyspark_code, results, steps))
            status_placeholder.empty()
        file_status_placeholder.empty()
        info_placeholder.empty()
        status_placeholder.success("Batch conversion complete!")
        st.session_state["conversion_in_progress"] = False  # Unlock UI

        # Store results in session state for persistence
        st.session_state["conversion_results"] = conversion_results

    # --- Results display (always use session_state) ---
    conversion_results = st.session_state.get("conversion_results", [])
    if conversion_results:
        result_file_names = [fname for fname, _, _, _, _ in conversion_results]
        selected_result_idx = st.selectbox(
            "Select a converted file to view results:",
            options=range(len(result_file_names)),
            format_func=lambda i: result_file_names[i],
            key="result_file_selectbox"
        )
        if 0 <= selected_result_idx < len(conversion_results):
            fname, outpath, code, results, steps = conversion_results[selected_result_idx]
            with st.expander(f"Results for {fname}", expanded=False):
                st.write(f"**Saved to:** `{outpath}`")
                st.write(f"**Final Conversion Judgement:** {results['judgement']}")
                st.write(f"**Iterations completed:** {results['iterations']}")
                st.write(f"**Agent Steps:** {' → '.join(steps)}")
                st.write(f"**Final Feedback:** \n{results['reflection'][-1] if results['reflection'] else 'No feedback provided'}")
                st.write("**Generated Code:**\n")
                st.code(code, language="python")
                st.download_button(
                    label=f"Download {fname}_converted.py",
                    data=code,
                    file_name=f"{os.path.splitext(fname)[0]}_converted.py",
                    mime="text/x-python"
                )
        else:
            st.info("No valid result selected.")
    else:
        st.info("No conversion results to display.")
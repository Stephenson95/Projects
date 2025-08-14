import re
import uuid
from graph import CodeConversionWorkflow

def main():

    print("Starting Code Conversion Workflow...")
    print("=" * 50)

    output_path = "./outputs"
    file_name = "output"
    workflow = CodeConversionWorkflow(max_iterations=3)
    
    # Example original code to convert
    with open("data/testdata4.md", "r", encoding="utf-8") as f:
        datastage_xml = f.read()
    
    work_id = str(uuid.uuid4())
    results = workflow.run(datastage_xml, thread_id = work_id)

    with open(output_path + f'/{file_name}.py', "w", encoding="utf-8") as f:
        # Removing code block markers
        f.write(re.sub(r"^```python\s*|```$", "", results['generation'][-1], flags=re.MULTILINE))

    print("Final Results:")
    print(f"Final Conversion Judgement: {results['judgement']}")
    print(f"Iterations completed: {results['iterations']}")
    print(f"Final PySpark code:\n{results['generation'][-1]}")
    print(f"Final Feedback:\n{results['reflection'][-1] if results['reflection'] else 'No feedback provided'}")
    
    print("\n" + "=" * 50)
    print(f"\nFinal PySpark code saved to: {output_path}")


    #Streaming example
    # for i, event in enumerate(workflow.stream_workflow(datastage_xml, thread_id = "conversion_001")):
    #     print(f"Step {i+1}: {list(event.keys())}")
    #     if "reflector" in event:
    #         state = event
    #         print(f"  Iteration: {state.get('iterations', 'N/A')}")
    #         if state.get("reflection"):
    #             print(f"  Latest feedback: {state['reflection'][-1][:100]}...")

if __name__ == "__main__":
    main()
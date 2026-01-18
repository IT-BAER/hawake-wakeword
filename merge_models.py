import onnx
from onnx import helper
import os
import sys

def merge_onnx_models(embedding_model_path, custom_model_path, output_path):
    print(f"Merging models:\n  Embedding: {embedding_model_path}\n  Custom: {custom_model_path}")
    
    # Load models
    emb_model = onnx.load(embedding_model_path)
    cust_model = onnx.load(custom_model_path)
    
    # Prefixing to avoid name collisions is usually good practice, 
    # but openwakeword models usually align inputs/outputs.
    
    # Embedding output name: usually 'output_1' or similar
    # Custom input name: usually 'input_1' or similar
    # We need to connect them.
    
    emb_output = emb_model.graph.output[0].name
    cust_input = cust_model.graph.input[0].name
    
    print(f"  Connecting {emb_output} -> {cust_input}")
    
    # Rename custom model input to match embedding model output
    for node in cust_model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == cust_input:
                node.input[i] = emb_output
    
    # Create merged graph
    # We take the inputs from embedding model and outputs from custom model
    
    # Combine nodes and initializers
    merged_nodes = list(emb_model.graph.node) + list(cust_model.graph.node)
    merged_initializers = list(emb_model.graph.initializer) + list(cust_model.graph.initializer)
    # merged_value_info = list(emb_model.graph.value_info) + list(cust_model.graph.value_info)
    
    # Fix output names collisions in initializers if any? 
    # Usually minimal risk with these specific models.
    
    merged_graph = helper.make_graph(
        nodes=merged_nodes,
        name="merged_openwakeword_model",
        inputs=emb_model.graph.input,
        outputs=cust_model.graph.output,
        initializer=merged_initializers
    )
    
    merged_model = helper.make_model(merged_graph, opset_imports=emb_model.opset_import)
    
    # CRITICAL: Set IR version to 7 for ONNX Runtime Android 1.14.0 compatibility
    # ONNX Runtime Android 1.14.0 only supports IR version <= 8
    merged_model.ir_version = 7
    print(f"  Set IR version to 7 for Android compatibility")
    
    onnx.save(merged_model, output_path)
    print(f"Successfully merged to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python merge_models.py <embedding_model> <custom_model> <output_model>")
        sys.exit(1)
        
    merge_onnx_models(sys.argv[1], sys.argv[2], sys.argv[3])

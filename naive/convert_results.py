import json
import os

def parse_results_to_json(content):
    results = {
        "overall_metrics": {},
        "combined": {},
        "individual": {}
    }
    
    current_section = None
    current_category = None
    
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and headers
        if not line or line.startswith("===") or line.endswith("==="):
            continue
            
        # Detect sections
        if line == "Overall Metrics:":
            current_section = "overall"
            continue
        elif line == "Metrics by Reasoning Type:":
            current_section = "metrics"
            continue
        elif line in ["combined:", "individual:"]:
            current_category = line[:-1]
            continue
            
        # Parse metrics
        if current_section == "overall":
            if line.startswith("accuracy:") or line.startswith("mean_similarity:"):
                key, value = line.split(":")
                results["overall_metrics"][key.strip()] = float(value.strip())
        elif current_section == "metrics" and current_category:
            if line.startswith("  "):  # This is a reasoning type line
                try:
                    reasoning_type, metrics_str = line.strip().split(": ", 1)
                    metrics_dict = eval(metrics_str)  # Convert string dict to actual dict
                    results[current_category][reasoning_type] = metrics_dict
                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error details: {str(e)}")
                    continue
    
    return results

def convert_results():
    # Convert both Gemini and GPT results
    for model in ["gemini", "gpt"]:
        input_file = f"results/{model}_results.txt"
        output_file = f"results/{model}_results.json"
        
        print(f"\nConverting {input_file} to JSON...")
        
        try:
            # Read the input file
            with open(input_file, "r") as f:
                content = f.read()
            
            # Parse and convert to JSON
            results = parse_results_to_json(content)
            
            # Write JSON output
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Successfully converted {input_file} to {output_file}")
            
        except Exception as e:
            print(f"Error converting {input_file}: {str(e)}")

if __name__ == "__main__":
    convert_results()

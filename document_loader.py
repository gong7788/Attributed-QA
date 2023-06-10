import json

data_path = 'data/doc2dial/'
file_path = data_path + 'doc2dial_doc.json'
with open(file_path, 'r') as f:
    data = json.load(f)

def filter_and_write_to_json(data, keys, output_file) -> None:
    # Create a new dictionary with only the desired keys
    filtered_data = {key: data[key] for key in keys if key in data}

    # Write the filtered data to a JSON file
    with open(output_file, 'w') as file:
        json.dump(filtered_data, file)

doc_list = list(data['doc_data']['ssa'].keys())

keys_to_filter = ['title', 'doc_text']
filter_and_write_to_json(data['doc_data']['ssa'][doc_list[0]], keys_to_filter, 'data/doc1.json')
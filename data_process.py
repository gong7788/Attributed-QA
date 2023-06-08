import json

data_path = "train1.json"

with open(data_path, 'r') as f:
    data = json.load(f)

def clean_data(text):
    text = text.replace('{vocalsound}', '')
    text = text.replace('{disfmarker}', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('T_V_', 'TV')
    text = text.replace('{pause}', '')
    text = text.replace('{nonvocalsound}', '')
    text = text.replace('{gap}', '')
    return text

#TODO multiple json files case
def extract_text_from_json(json_data):
    data = []
    for turn, meeting in enumerate(json_data['meeting_transcripts']):
        if meeting:
            data.append({
                'turn': turn,
                'speaker': meeting['speaker'],
                'content': clean_data(meeting['content']),
            })
    return data

temp = extract_text_from_json(data)

with open("data.json", "w") as f:
    json.dump(temp, f)
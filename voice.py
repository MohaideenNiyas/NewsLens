# First, let's save your script to a file
with open("kabul_report.txt", "w") as f:
    f.write("""--- AI NEWS REPORTER SCRIPT --- [ Sound of gunfire in the distance, followed by the revving of a motorcycle ] I'm Rania Rashid, reporting live from the streets of Kabul, where the fragile calm is being shattered by the Taliban's defiant rejection of the International Criminal Court's arrest warrants for two of its top officials. [ Sounds of chaos in the background, people shouting ] According to eyewitnesses, the Taliban is refusing to comply with the ICC's request to arrest Hibatullah Akhundzada, the leader, and another top official, accused of perpetrating human rights abuses against women. The ICC's chief prosecutor, Karim Khan, announced the move yesterday, citing the Taliban's brutal crackdown on women's rights since taking control of the country in 2021. [ Pause ] Since then, women have been barred from most public spaces, education beyond sixth grade, and even the workforce. It's a stark reminder of the Taliban's draconian rule, and the world is watching with bated breath as this crisis unfolds. [ Sounds of helicopters overhead, sirens in the distance ] The foreign ministry has condemned the ICC's move, accusing it of political bias and interference in Afghanistan's internal affairs. But human rights groups are calling for the ICC to stand firm, demanding justice for the countless women and girls who have suffered under the Taliban's rule. [ Sounds of shouting and gunfire getting closer ] As I speak, the situation is escalating rapidly. The Taliban is mobilizing its forces, and the streets are filling with protesters. The international community is urging calm, but the stakes couldn't be higher. Will the ICC's bold move spark a new era of accountability, or will it ignite a powder keg of conflict? [ Pause ] I'm Rania Rashid, reporting from Kabul. Stay safe, stay tuned. [ Sign-off music plays ] "From the frontlines to your living room, reporting the world's most critical stories – that's what I do." --------------------------------------------------""")

# Initialize the pipeline with the API key
from zyphra import ZyphraClient
import os
import base64
from IPython.display import Audio
from typing import Dict, Optional, Union, List

# Create the pipeline instance
pipeline = NewsToSpeechPipeline(api_key='zsk-f97f665353e18ab8620736aeb0c1c43cb4034ec6f41723a31cbbd337b378cae3')

# Process the file
processed_file_path = process_report_from_file("kabul_report.txt", "kabul_report.mp3")
print(f"Processed Kabul report at: {processed_file_path}")

# Let's also create a segmented version for more control over different parts of the report
# First, let's clean up the script to remove sound effects notes in brackets
def clean_script(script):
    import re
    # Remove content within square brackets (sound effects)
    cleaned = re.sub(r'\[.*?\]', '', script)
    return cleaned

# Get the clean script
with open("kabul_report.txt", 'r') as file:
    content = file.read()
    if "--- AI NEWS REPORTER SCRIPT ---" in content:
        parts = content.split("--- AI NEWS REPORTER SCRIPT ---")
        if len(parts) > 1:
            raw_script = parts[1].strip()
            if "----------" in raw_script:
                raw_script = raw_script.split("----------")[0].strip()
        else:
            raw_script = content
    else:
        raw_script = content

# Clean the script
clean_report = clean_script(raw_script)

# Now break it into segments
segments = [
    {
        "text": "I'm Rania Rashid, reporting live from the streets of Kabul, where the fragile calm is being shattered by the Taliban's defiant rejection of the International Criminal Court's arrest warrants for two of its top officials.",
        "speaking_rate": 15.0,
        "emotion_settings": {"neutral": 0.6, "surprise": 0.3, "fear": 0.1},
        "filename": "kabul_intro.mp3",
        "mime_type": "audio/mp3"
    },
    {
        "text": "According to eyewitnesses, the Taliban is refusing to comply with the ICC's request to arrest Hibatullah Akhundzada, the leader, and another top official, accused of perpetrating human rights abuses against women. The ICC's chief prosecutor, Karim Khan, announced the move yesterday, citing the Taliban's brutal crackdown on women's rights since taking control of the country in 2021.",
        "speaking_rate": 14.5,
        "emotion_settings": {"neutral": 0.7, "sadness": 0.2, "fear": 0.1},
        "filename": "kabul_details.mp3",
        "mime_type": "audio/mp3"
    },
    {
        "text": "Since then, women have been barred from most public spaces, education beyond sixth grade, and even the workforce. It's a stark reminder of the Taliban's draconian rule, and the world is watching with bated breath as this crisis unfolds.",
        "speaking_rate": 14.0,
        "emotion_settings": {"neutral": 0.5, "sadness": 0.4, "anger": 0.1},
        "filename": "kabul_impact.mp3",
        "mime_type": "audio/mp3"
    },
    {
        "text": "The foreign ministry has condemned the ICC's move, accusing it of political bias and interference in Afghanistan's internal affairs. But human rights groups are calling for the ICC to stand firm, demanding justice for the countless women and girls who have suffered under the Taliban's rule.",
        "speaking_rate": 14.5,
        "emotion_settings": {"neutral": 0.6, "anger": 0.2, "sadness": 0.2},
        "filename": "kabul_reactions.mp3",
        "mime_type": "audio/mp3"
    },
    {
        "text": "As I speak, the situation is escalating rapidly. The Taliban is mobilizing its forces, and the streets are filling with protesters. The international community is urging calm, but the stakes couldn't be higher. Will the ICC's bold move spark a new era of accountability, or will it ignite a powder keg of conflict?",
        "speaking_rate": 15.0,
        "emotion_settings": {"neutral": 0.5, "fear": 0.3, "surprise": 0.2},
        "filename": "kabul_escalation.mp3",
        "mime_type": "audio/mp3"
    },
    {
        "text": "I'm Rania Rashid, reporting from Kabul. Stay safe, stay tuned. From the frontlines to your living room, reporting the world's most critical stories – that's what I do.",
        "speaking_rate": 13.5,
        "emotion_settings": {"neutral": 0.8, "sadness": 0.1, "fear": 0.1},
        "filename": "kabul_signoff.mp3",
        "mime_type": "audio/mp3"
    }
]

# Process the segments
segment_paths = pipeline.process_script_segments(
    segments=segments,
    output_dir="kabul_report_segments",
    language_iso_code="en-us",
    model="zonos-v0.1-transformer"
)

print(f"Generated {len(segment_paths)} segmented Kabul report audio files")

# Play the generated audio
print("\nPlaying the full Kabul report:")
display(Audio(processed_file_path))

print("\nPlaying the first segment (intro):")
display(Audio(os.path.join("kabul_report_segments", "kabul_intro.mp3")))
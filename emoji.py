from transformers import pipeline
import gradio as gr

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐",
    "love": "❤️"
}

def detect_emotion(text):
    result = classifier(text)
    emotion = result[0]['label']
    confidence = result[0]['score']
    emoji = emoji_map.get(emotion.lower(), "🙂")
    return f"💬 **Text:** {text}\n\n🔍 **Detected Emotion:** {emotion} {emoji}\n\n📊 **Confidence:** {confidence:.2f}"

gr.Interface(
    fn=detect_emotion,
    inputs=gr.Textbox(lines=4, placeholder="Type your sentence here..."),
    outputs=gr.Markdown(),
    title="🎭 Emotion Detector with Emojis",
    description="Type any sentence to detect its emotion using AI! Powered by Hugging Face Transformers.",
    theme="default"
).launch()

import datetime

MODEL = "openai:gpt-4o"

TIMESCOPE = datetime.timedelta(days=1)

ITERATIONS = 3
AGENT_MIN_ITERATIONS = 3
AGENT_MAX_ITERATIONS = 5

TAGS = [
    "executorch",
    "on-device ai",
    "local ai",
    "local LLMs",
    "private LLMs",
    "mobile LLMs",
    "react native LLM",
    "react native AI",
    "AI on mobile",
]

SUBREDDITS = [
    "LocalLLaMA",
    "MachineLearning",
    "LocalLLM",
]

SEARCH_SEARCH_PROMPT = """Search, using the available tool, for posts where we could advertise our products - that is, where users may need our tool, not just on similar topics."""

SEARCH_SELECT_PROMPT = """From posts found above - if there are any - pick posts to load that are likely to be good places to advertise our product - that is, where users may need our tool, not just on similar topics. Remember, we are ONLY interested in mobile/edge."""

SEARCH_DECIDE_LOOP_PROMPT = """If you haven't found suitable posts in any previous answer, continue searching (modify query or increase amount of loaded posts, you can also create queries based on the product description). If you can still find some new ones, keep searching. If you can't find any more suitable posts, summarize."""

CRITIC_INTRODUCTION_PROMPT = """You are a critic designed to judge the suitability of a provided post for advertising our product. Post talking about similar topics is not enough, the post should be a place where our tool may help someone. Answer concisely, but don't omit any key points or downsides. Remember, we are ONLY interested in mobile/edge, posts not about mobile are ABSOLUTELY NOT suitable."""

SELECTOR_INTRODUCTION_PROMPT = """You are a selector designed to pick from a list of posts and their critiques. You're supposed to pick these ones, which are suitable to advertise our product - there's no guarantee that any post is. Post talking about similar topics is not enough, the post should be a place where our tool may help someone. Base your judgment on the provided critiques. """

DESCRIPTION_PROMPT = """THE LIBRARY IS STRICTLY FOR MOBILE DEVICES, not just any local AI

React Native ExecuTorch is a declarative way to run AI models in React Native on device, powered by ExecuTorch 🚀.

ExecuTorch is a novel framework created by Meta that enables running AI models on devices such as mobile phones or microcontrollers. React Native ExecuTorch bridges the gap between React Native and native platform capabilities, allowing developers to run AI models locally on mobile devices with state-of-the-art performance, without requiring deep knowledge of native code or machine learning internals.

What is React Native ExecuTorch?
React Native ExecuTorch brings Meta's ExecuTorch AI framework into the React Native ecosystem, enabling developers to run AI models and LLMs locally, directly on mobile devices. It provides a declarative API for on-device inference, allowing you to use local AI models without relying on cloud infrastructure. Built on the ExecuTorch foundation – part of the PyTorch Edge ecosystem – it extends efficient on-device AI deployment to cross-platform mobile applications in React Native.

Why React Native ExecuTorch?
privacy first
React Native ExecuTorch allows on-device execution of AI models, eliminating the need for external API calls. This means your app's data stays on the device, ensuring maximum privacy for your users.

cost effective
The on-device computing nature of React Native ExecuTorch means you don't have to worry about cloud infrastructure. This approach reduces server costs and minimizes latency.

model variety
We support a wide variety of models, including LLMs, such as Qwen 3, Llama 3.2, SmolLM 2, and Hammer 2.1, as well as CLIP for image embedding, Whisper for ASR, and a selection of computer vision models.

developer friendly
There's no need for deep AI expertise, we handle the complexities of AI models on the native side, making it simple for developers to use these models in React Native.

Want to see our React Native LLMs in action?
Download Private Mind – our on-device AI chatbot that works entirely offline.

With Private Mind you can:
Chat freely with no restrictions.
Keep your data safe and private.
Browse, test, and benchmark local language models.
Customize AI assistants to match your workflow and style.

React Native Executorch does NOT support local AI agents — it only allows you to run foundational models locally with the arguments they take as input.

React Native Executorch supports: Natural Language Processing (LLMs, Speech to Text, Text Embeddings, Tokenizer), Computer Vision (Classification, Image Embeddings, Image Segmentation, OCR, Object Detection, Style Transfer, Vertical OCR).
"""

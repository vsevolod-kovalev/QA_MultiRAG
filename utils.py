import numpy as np
import math
import whisper
from moviepy import VideoFileClip
import torch
import faiss
from PIL import Image
import matplotlib.pyplot as plt

def frames_from_video(filepath):
    # video -> frames
    clip = VideoFileClip(filepath)
    print(f"Video Duration: {clip.duration} seconds")
    frames = []
    for t in range(0, math.floor(clip.duration)):
        frame = clip.get_frame(t)
        frames.append(frame)
    clip.close()
    return frames


def audio_from_video(input_path, output_path):
    # video -> None (audio saved in output_path)
    clip = VideoFileClip(input_path)
    audio = clip.audio
    audio.write_audiofile(output_path)
    clip.close()

def captions_from_audio(audio_inputpath):
    # audio -> text transcription
    model = whisper.load_model("base")
    return model.transcribe(audio_inputpath, task="transcribe")

def align_frames_captions(audio_inputpath, frames, overlap=5):
    transcription = captions_from_audio(audio_inputpath)
    num_frames = len(frames)
    alignments = [{'frame': frames[i], 'caption': []} for i in range(num_frames)]

    for segment in transcription['segments']:
        text = segment['text']
        start_time = max(0, math.ceil(segment['start']) - overlap)
        end_time = min(num_frames, math.ceil(segment['end']) + overlap)

        for frame_idx in range(start_time, end_time):
            alignments[frame_idx]['caption'].append(text)

    # Combine captions for each frame
    for frame in alignments:
        frame['caption'] = '\n'.join(frame['caption'])
    return alignments


def alignments_to_embeddings(alignments, model, processor):
    # Load CLIP model and processor

    embeddings = []  # To store combined embeddings for each alignment
    max_text_length = 77  # Maximum sequence length for CLIP

    for frame in alignments:
        image = frame['frame']
        caption = frame['caption']

        # Convert frame to PIL Image
        pil_image = Image.fromarray(image)
        if not caption:
            caption = ["No Caption"]

        # Truncate captions to the max length supported by CLIP
        truncated_caption = [c[:max_text_length] for c in caption]

        # Prepare inputs for model
        inputs = processor(text=truncated_caption, images=pil_image, return_tensors="pt", padding=True)

        # Separate image and text inputs
        image_inputs = inputs["pixel_values"]
        text_inputs = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Forward pass to get embeddings
        with torch.no_grad():
            image_emb = model.get_image_features(pixel_values=image_inputs)
            text_emb = model.get_text_features(input_ids=text_inputs, attention_mask=attention_mask)

        # Normalize embeddings
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Combine embeddings
        combined_emb = (image_emb + text_emb.mean(dim=0)) / 2

        # Append combined embedding
        embeddings.append(combined_emb.cpu().numpy())

    return np.vstack(embeddings)

# Store embeddings in FAISS
def store_embeddings(embeddings, dimension):
    # Create FAISS index for L2 similarity search
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Add embeddings to the index
    return index

def search_similar(query_embedding, index, k=3):
    # Ensure query embedding is 2D
    query_embedding = query_embedding.reshape(1, -1)

    # Perform the search
    distances, indices = index.search(query_embedding, k)
    return indices, distances


def search_similar_with_mmr(query_embedding, index, embeddings, k=3, lambda_param=0.5):
    # Ensure query embedding is 2D
    query_embedding = query_embedding.reshape(1, -1)

    # Perform initial search to get all potential matches
    distances, indices = index.search(query_embedding, len(embeddings))

    # Flatten results for processing
    distances = distances[0]
    indices = indices[0]

    # Initialize MMR variables
    selected = []
    candidate_indices = set(indices)

    # Iteratively select `k` items
    for _ in range(k):
        mmr_score = {}
        for candidate in candidate_indices:
            # Calculate relevance (similarity to query)
            relevance = 1 - distances[candidate]

            # Calculate diversity (similarity to selected items)
            if selected:
                diversity = max(
                    1 - np.linalg.norm(embeddings[candidate] - embeddings[s]) for s in selected
                )
            else:
                diversity = 0

            # MMR score
            mmr_score[candidate] = lambda_param * relevance - (1 - lambda_param) * diversity

        # Select the item with the highest MMR score
        best_candidate = max(mmr_score, key=mmr_score.get)
        selected.append(best_candidate)
        candidate_indices.remove(best_candidate)

    # Get distances for selected indices
    selected_distances = [distances[s] for s in selected]
    return selected, selected_distances

def text_to_embedding(text_query, model, processor):
    inputs = processor(text=[text_query], images=None, return_tensors="pt", padding=True)
    text_inputs = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        text_emb = model.get_text_features(input_ids=text_inputs, attention_mask=attention_mask)

    # Normalize the embedding
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy()

def query_llava(text_query, device, model, processor, frames, captions):
    combined_captions = " ".join(captions)[:1024]
    print(f"Combined frames:\t{combined_captions}\n")
    representative_frame = Image.fromarray(frames[0]).resize((224, 224))
    plt.imshow(representative_frame)
    plt.show()

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{text_query}. Context: {combined_captions}"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=representative_frame, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=50)
    
    response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return response
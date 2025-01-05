from utils import (
    frames_from_video,
    audio_from_video,
    align_frames_captions,
    alignments_to_embeddings,
    store_embeddings,
    search_similar_with_mmr,
    text_to_embedding,
    query_llava,
)
import torch
from PIL import Image

from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoProcessor,
    LlavaForConditionalGeneration
)


def main():
    # Load model with MPS support
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


    LLAVA_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", 
        torch_dtype=torch.float16
    ).to(device)
    LLAVA_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")


    CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    

    # LLAVA_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    # LLAVA_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    video_path = "CNN_lecture_shorter.mp4"
    audio_path = "buffer_audio.mp3"

    # Extract audio from the video
    audio_from_video(video_path, audio_path)

    # Extract frames from the video
    frames = frames_from_video(video_path)

    # Align frames with audio captions
    alignments = align_frames_captions(audio_path, frames)

    # Convert alignments to embeddings
    embeddings = alignments_to_embeddings(alignments, CLIP_model, CLIP_processor)

    # Store embeddings in FAISS index
    faiss_index = store_embeddings(embeddings, embeddings.shape[1])

    # Input textual query from the user
    text_query = input("Enter your textual query: ")

    # Convert the textual query to an embedding
    query_embedding = text_to_embedding(text_query, CLIP_model, CLIP_processor)

    # Search similar frames in the FAISS index with MMR
    indices, distances = search_similar_with_mmr(
        query_embedding=query_embedding,
        index=faiss_index,
        embeddings=embeddings,
        k=3,
        lambda_param=0.7,
    )

    # Display results
    print(f"Top {len(indices)} similar frames:")
    for idx, distance in zip(indices, distances):
        print(f"Frame {idx}: Caption -> {alignments[idx]['caption']}, Distance -> {distance}")

    retrieved_frames = [alignments[idx]['frame'] for idx in indices]
    retrieved_captions = [alignments[idx]['caption'] for idx in indices]
    response = query_llava(text_query, device, LLAVA_model, LLAVA_processor, retrieved_frames, retrieved_captions)
    print(response)
    # for i, response in enumerate(responses):
        # print(f"Response for frame {i}: {response}")

if __name__ == "__main__":
    main()

import argparse
import copy
import glob
import os
import pickle
import re
import shutil
import subprocess
import sys

import cv2
import gradio as gr
import imageio
import numpy as np
import torch
from argparse import Namespace
from moviepy.editor import AudioFileClip, VideoFileClip
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import (
    coord_placeholder,
    get_bbox_range,
    get_landmark_and_bbox,
    read_imgs,
)
from musetalk.utils.utils import datagen, get_file_type, get_video_fps, load_all_model

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "models")


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def check_required_models():
    required_models = {
        "MuseTalk V1.5 UNet": f"{CHECKPOINTS_DIR}/musetalkV15/unet.pth",
        "MuseTalk V1.5 config": f"{CHECKPOINTS_DIR}/musetalkV15/musetalk.json",
        "SD VAE": f"{CHECKPOINTS_DIR}/sd-vae/config.json",
        "Whisper": f"{CHECKPOINTS_DIR}/whisper/config.json",
        "DWPose": f"{CHECKPOINTS_DIR}/dwpose/dw-ll_ucoco_384.pth",
        "SyncNet": f"{CHECKPOINTS_DIR}/syncnet/latentsync_syncnet.pt",
        "Face Parse": f"{CHECKPOINTS_DIR}/face-parse-bisent/79999_iter.pth",
        "ResNet": f"{CHECKPOINTS_DIR}/face-parse-bisent/resnet18-5c106cde.pth",
    }
    missing = [name for name, path in required_models.items() if not os.path.exists(path)]
    if missing:
        print("다음 필수 모델 파일이 없습니다:")
        for name in missing:
            print(f"- {name}")
        print("\n누락된 모델을 받으려면 다운로드 스크립트를 실행하세요.")
        print("Windows: download_weights.bat" if sys.platform == "win32" else "Linux/macOS: ./download_weights.sh")
        sys.exit(1)


def check_video(video):
    if not isinstance(video, str):
        return video
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video

    os.makedirs("./results/input", exist_ok=True)
    output_video = os.path.join("./results/input", "outputxxx_" + file_name)

    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()["fps"]
    frames = [im for im in reader]
    reader.close()

    target_fps = 25
    target_length = int(len(frames) / fps * target_fps)
    original_t = [x / fps for x in range(1, len(frames) + 1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, target_length + 1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1
            if t_idx >= len(frames):
                t_idx = len(frames) - 1
                break
        target_frames.append(frames[t_idx])

    imageio.mimwrite(output_video, target_frames, "FFMPEG", fps=25, codec="libx264", quality=9, pixelformat="yuv420p")
    return output_video


@torch.no_grad()
def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", left_cheek_width=90, right_cheek_width=90):
    os.makedirs("./results/debug", exist_ok=True)

    if get_file_type(video_path) == "video":
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.imread(video_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    debug_frame_path = "./results/debug/debug_frame.png"
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox, frame = coord_list[0], frame_list[0]

    if bbox == coord_placeholder:
        return None, "얼굴을 찾지 못했습니다. bbox_shift 값을 조정해 보세요."

    fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
    x1, y1, x2, y2 = bbox
    y2 = min(y2 + extra_margin, frame.shape[0])
    crop_frame = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)

    random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    latents = vae.get_latents_for_unet(crop_frame).to(dtype=weight_dtype)
    pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)

    res_frame = cv2.resize(recon[0].astype(np.uint8), (x2 - x1, y2 - y1))
    combined = get_image(frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=fp)

    info = (
        f"bbox_shift: {bbox_shift}\n"
        f"extra_margin: {extra_margin}\n"
        f"parsing_mode: {parsing_mode}\n"
        f"left_cheek_width: {left_cheek_width}\n"
        f"right_cheek_width: {right_cheek_width}\n"
        f"bbox: [{x1}, {y1}, {x2}, {y2}]"
    )
    return cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), info


@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", left_cheek_width=90, right_cheek_width=90, progress=gr.Progress(track_tqdm=True)):
    args_local = Namespace(
        result_dir="./results/output",
        fps=25,
        batch_size=8,
        use_saved_coord=False,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        version="v15",
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
    )

    input_basename = os.path.basename(video_path).split(".")[0]
    audio_basename = os.path.basename(audio_path).split(".")[0]
    output_basename = f"{input_basename}_{audio_basename}"

    temp_dir = os.path.join(args_local.result_dir, args_local.version)
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args_local.result_dir, "..", input_basename + ".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)
    output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")

    if get_file_type(video_path) == "video":
        save_dir = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir, exist_ok=True)
        reader = imageio.get_reader(video_path)
        for i, frame in enumerate(reader):
            imageio.imwrite(f"{save_dir}/{i:08d}.png", frame)
        reader.close()
        input_img_list = sorted(glob.glob(os.path.join(save_dir, "*.[jpJP][pnPN]*[gG]")))
        fps = get_video_fps(video_path)
    else:
        input_img_list = sorted(glob.glob(os.path.join(video_path, "*.[jpJP][pnPN]*[gG]")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args_local.fps

    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=fps,
        audio_padding_length_left=args_local.audio_padding_length_left,
        audio_padding_length_right=args_local.audio_padding_length_right,
    )

    if os.path.exists(crop_coord_save_path) and args_local.use_saved_coord:
        with open(crop_coord_save_path, "rb") as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, "wb") as f:
            pickle.dump(coord_list, f)

    bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)

    input_latents = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + extra_margin, frame.shape[0])
        crop_frame = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        input_latents.append(vae.get_latents_for_unet(crop_frame))

    frame_cycle = frame_list + frame_list[::-1]
    coord_cycle = coord_list + coord_list[::-1]
    latent_cycle = input_latents + input_latents[::-1]

    generated = []
    gen = datagen(whisper_chunks=whisper_chunks, vae_encode_latents=latent_cycle, batch_size=args_local.batch_size, delay_frame=0, device=device)
    total = int(np.ceil(float(len(whisper_chunks)) / args_local.batch_size))
    for whisper_batch, latent_batch in tqdm(gen, total=total):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        generated.extend(vae.decode_latents(pred_latents))

    for i, res_frame in enumerate(tqdm(generated)):
        bbox = coord_cycle[i % len(coord_cycle)]
        frame = copy.deepcopy(frame_cycle[i % len(frame_cycle)])
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + extra_margin, frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:
            continue
        combined = get_image(frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=fp)
        cv2.imwrite(f"{result_img_save_path}/{i:08d}.png", combined)

    files = [f for f in os.listdir(result_img_save_path) if re.match(r"\d{8}\.png", f)]
    files.sort(key=lambda x: int(x.split(".")[0]))
    frames = [imageio.imread(os.path.join(result_img_save_path, f)) for f in files]
    imageio.mimwrite("temp.mp4", frames, "FFMPEG", fps=25, codec="libx264", pixelformat="yuv420p")

    video_clip = VideoFileClip("temp.mp4")
    audio_clip = AudioFileClip(audio_path)
    video_clip.set_audio(audio_clip).write_videofile(output_vid_name, codec="libx264", audio_codec="aac", fps=25)
    os.remove("temp.mp4")

    return output_vid_name, bbox_shift_text


parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_path", type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin")
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
parser.add_argument("--use_float16", action="store_true")
args = parser.parse_args()

if not fast_check_ffmpeg():
    separator = ";" if sys.platform == "win32" else ":"
    os.environ["PATH"] = f"{args.ffmpeg_path}{separator}{os.environ['PATH']}"

check_required_models()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device,
)

weight_dtype = torch.float16 if args.use_float16 else torch.float32
if args.use_float16:
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()

pe = pe.to(device)
vae.vae = vae.vae.to(device)
unet.model = unet.model.to(device)
timesteps = torch.tensor([0], device=device)

audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper").to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

with gr.Blocks(title="Audio-Driven Talking-Head Generation") as demo:
    gr.Markdown("# Audio-Driven Talking-Head Generation\nMuseTalk-based lip-sync generation demo")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="음성 입력", type="filepath")
            video = gr.Video(label="레퍼런스 영상", sources=["upload"])
            bbox_shift = gr.Number(label="얼굴 박스 이동값 (px)", value=0)
            parsing_mode = gr.Radio(label="파싱 모드", choices=["jaw", "raw"], value="jaw")
            extra_margin = gr.Slider(label="추가 마진", minimum=0, maximum=40, value=10, step=1)
            left_cheek_width = gr.Slider(label="왼쪽 볼 범위", minimum=20, maximum=160, value=90, step=5)
            right_cheek_width = gr.Slider(label="오른쪽 볼 범위", minimum=20, maximum=160, value=90, step=5)
            debug_btn = gr.Button("1. 인페인팅 테스트")
            generate_btn = gr.Button("2. 생성하기", variant="primary")
        with gr.Column():
            debug_image = gr.Image(label="인페인팅 테스트 결과")
            debug_info = gr.Textbox(label="파라미터 정보", lines=6)
            output_video = gr.Video(label="생성 결과")
            bbox_info = gr.Textbox(label="BBox 안내")

    video.change(fn=check_video, inputs=[video], outputs=[video])
    debug_btn.click(
        fn=debug_inpainting,
        inputs=[video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
        outputs=[debug_image, debug_info],
    )
    generate_btn.click(
        fn=inference,
        inputs=[audio, video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
        outputs=[output_video, bbox_info],
    )

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

demo.queue().launch(share=args.share, debug=True, server_name=args.ip, server_port=args.port)

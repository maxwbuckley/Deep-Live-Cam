import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional
import insightface
import threading

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path

FACE_ANALYSER = None
FACE_ANALYSER_LOCK = threading.Lock()

# Thread pool for running landmark + recognition in parallel
_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def get_face_analyser() -> Any:
    """Get face analyser with thread-safe initialization."""
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        with FACE_ANALYSER_LOCK:
            # Double-check after acquiring lock
            if FACE_ANALYSER is None:
                from modules.processors.frame._onnx_enhancer import (
                    build_provider_config,
                )
                providers = build_provider_config()
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=providers,
                    allowed_modules=['detection', 'recognition', 'landmark_2d_106']
                )
                FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def _needs_landmark() -> bool:
    """Check whether any active feature requires 106-point landmarks.

    Landmarks are needed by face enhancers and mouth masking, but not
    by the face swapper alone.
    """
    if getattr(modules.globals, "mouth_mask", False):
        return True
    fp_ui = getattr(modules.globals, "fp_ui", {})
    for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
        if fp_ui.get(key, False):
            return True
    processors = getattr(modules.globals, "frame_processors", [])
    for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
        if key in processors:
            return True
    return False


def _is_dml() -> bool:
    return any("DmlExecutionProvider" in p for p in modules.globals.execution_providers)


def _analyse_faces(frame: Frame) -> list:
    """Run face detection then landmark + recognition, parallelising where possible.

    InsightFace's default ``FaceAnalysis.get()`` runs all post-detection
    models sequentially.  Landmark and recognition are independent of each
    other (both only need the detection bbox/kps), so we run them in
    parallel.  When landmarks aren't needed (swap-only pipeline), we skip
    that model entirely.
    """
    fa = get_face_analyser()

    # --- 1. Detection (always required) ---
    bboxes, kpss = fa.det_model.detect(frame, max_num=0, metric="default")
    if bboxes.shape[0] == 0:
        return []

    need_landmark = _needs_landmark()

    # Look up post-detection models once
    rec_model = fa.models.get("recognition")
    lmk_model = fa.models.get("landmark_2d_106") if need_landmark else None

    # --- 2. Build Face objects and run post-detection models ---
    from insightface.app.common import Face

    faces = []
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4], kps=kpss[i] if kpss is not None else None, det_score=bboxes[i, 4])
        faces.append(face)

    for face in faces:
        if lmk_model is not None and rec_model is not None:
            # Run landmark and recognition in parallel — they are
            # independent and use different ONNX sessions.
            lmk_future = _MODEL_EXECUTOR.submit(lmk_model.get, frame, face)
            rec_future = _MODEL_EXECUTOR.submit(rec_model.get, frame, face)
            lmk_future.result()
            rec_future.result()
        elif rec_model is not None:
            rec_model.get(frame, face)
        elif lmk_model is not None:
            lmk_model.get(frame, face)

    return faces


def get_one_face(frame: Frame) -> Any:
    if _is_dml():
        with modules.globals.dml_lock:
            faces = _analyse_faces(frame)
    else:
        faces = _analyse_faces(frame)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        if _is_dml():
            with modules.globals.dml_lock:
                return _analyse_faces(frame)
        else:
            return _analyse_faces(frame)
    except IndexError:
        return None

def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False

def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])

    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

        modules.globals.source_target_map.append({
                'id' : max_id + 1
                })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            modules.globals.source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []
    
        print('Creating temp resources...')
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print('Extracting frames...')
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

        i = 0
        for temp_frame_path in tqdm(temp_frame_paths, desc="Extracting face embeddings from frames"):
            temp_frame = cv2.imread(temp_frame_path)
            many_faces = get_many_faces(temp_frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)
            
            frame_face_embeddings.append({'frame': i, 'faces': many_faces, 'location': temp_frame_path})
            i += 1

        centroids = find_cluster_centroids(face_embeddings)

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index

        for i in range(len(centroids)):
            modules.globals.source_target_map.append({
                'id' : i
            })

            temp = []
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                temp.append({'frame': frame['frame'], 'faces': [face for face in frame['faces'] if face['target_centroid'] == i], 'location': frame['location']})

            modules.globals.source_target_map[i]['target_faces_in_frame'] = temp

        # dump_faces(centroids, frame_face_embeddings)
        default_target_face()
    except ValueError:
        return None
    

def default_target_face():
    for map in modules.globals.source_target_map:
        best_face = None
        best_frame = None
        for frame in map['target_faces_in_frame']:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        for frame in map['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame

        x_min, y_min, x_max, y_max = best_face['bbox']

        target_frame = cv2.imread(best_frame['location'])
        map['target'] = {
                        'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                        'face' : best_face
                        }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        cv2.imwrite(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1

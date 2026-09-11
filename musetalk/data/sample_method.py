import numpy as np
import random

def summarize_tensor(x):
    return f"\033[34m{str(tuple(x.shape)).ljust(24)}\033[0m (\033[31mmin {x.min().item():+.4f}\033[0m / \033[32mmean {x.mean().item():+.4f}\033[0m / \033[33mmax {x.max().item():+.4f}\033[0m)"

def calculate_mouth_open_similarity(landmarks_list, select_idx,top_k=50,ascending=True):
    num_landmarks = len(landmarks_list)
    mouth_open_ratios = np.zeros(num_landmarks)
    print(np.shape(landmarks_list))
    for i, landmarks in enumerate(landmarks_list):
        mouth_top = landmarks[165]
        mouth_bottom = landmarks[147]
        mouth_open_ratio = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_open_ratios[i] = mouth_open_ratio
    differences_matrix = np.abs(mouth_open_ratios[:, np.newaxis] - mouth_open_ratios[select_idx])
    differences_matrix_with_signs = mouth_open_ratios[:, np.newaxis] - mouth_open_ratios[select_idx]
    print(differences_matrix.shape)
    if ascending:
        top_indices = np.argsort(differences_matrix[i])[:top_k]
    else:
        top_indices = np.argsort(-differences_matrix[i])[:top_k]
    similar_landmarks_indices = top_indices.tolist()
    similar_landmarks_distances = differences_matrix_with_signs[i].tolist()
    return similar_landmarks_indices, similar_landmarks_distances

def get_closed_mouth(landmarks_list,ascending=True,top_k=50):
    num_landmarks = len(landmarks_list)
    mouth_open_ratios = np.zeros(num_landmarks)
    for i, landmarks in enumerate(landmarks_list):
        mouth_top = np.array(landmarks[165])
        mouth_bottom = np.array(landmarks[147])
        mouth_open_ratio = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_open_ratios[i] = mouth_open_ratio
    if ascending:
        top_indices = np.argsort(mouth_open_ratios)[:top_k]
    else:
        top_indices = np.argsort(-mouth_open_ratios)[:top_k]
    return top_indices

def calculate_landmarks_similarity(selected_idx, landmarks_list,image_shapes, start_index, end_index, top_k=50,ascending=True):
    num_landmarks = len(landmarks_list)
    resized_landmarks = []
    for i in range(num_landmarks):
        landmark_array = np.array(landmarks_list[i])
        selected_landmarks = landmark_array[start_index:end_index]
        resized_landmark = resize_landmark(selected_landmarks, w=image_shapes[i][0], h=image_shapes[i][1],new_w=256,new_h=256)
        resized_landmarks.append(resized_landmark)
    resized_landmarks_array = np.array(resized_landmarks)
    distances = np.linalg.norm(resized_landmarks_array - resized_landmarks_array[selected_idx][np.newaxis, :], axis=2)
    overall_distances = np.mean(distances, axis=1)
    if ascending:
        sorted_indices = np.argsort(overall_distances)
        similar_landmarks_indices = sorted_indices[1:top_k+1].tolist()
    else:
        sorted_indices = np.argsort(-overall_distances)
        similar_landmarks_indices = sorted_indices[0:top_k].tolist()
    return similar_landmarks_indices

def process_bbox_musetalk(face_array, landmark_array):
    x_min_face, y_min_face, x_max_face, y_max_face = map(int, face_array)
    x_min_lm = min([int(x) for x, y in landmark_array])
    y_min_lm = min([int(y) for x, y in landmark_array])
    x_max_lm = max([int(x) for x, y in landmark_array])
    y_max_lm = max([int(y) for x, y in landmark_array])
    x_min = min(x_min_face, x_min_lm)
    y_min = min(y_min_face, y_min_lm)
    x_max = max(x_max_face, x_max_lm)
    y_max = max(y_max_face, y_max_lm)
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    return [x_min, y_min, x_max, y_max]

def shift_landmarks_to_face_coordinates(landmark_list, face_list):
    landmark_list_shift = []
    bbox_union = []
    face_shapes = []
    for i in range(len(face_list)):
        landmark_array = np.array(landmark_list[i])
        face_array = face_list[i]
        f_landmark_bbox = process_bbox_musetalk(face_array, landmark_array)
        x_min, y_min, x_max, y_max = f_landmark_bbox
        landmark_array[:, 0] = landmark_array[:, 0] - f_landmark_bbox[0]
        landmark_array[:, 1] = landmark_array[:, 1] - f_landmark_bbox[1]
        landmark_list_shift.append(landmark_array)
        bbox_union.append(f_landmark_bbox)
        face_shapes.append((x_max - x_min, y_max - y_min))
    return landmark_list_shift, bbox_union, face_shapes

def resize_landmark(landmark, w, h, new_w, new_h):
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def get_src_idx(drive_idx, T, sample_method,landmarks_list,image_shapes,top_k_ratio):
    if sample_method == "random":
        src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)
    elif sample_method == "pose_similarity":
        top_k = int(top_k_ratio*len(landmarks_list))
        try:
            top_k = int(top_k_ratio*len(landmarks_list))
            landmark_start_idx = 0
            landmark_end_idx = 16
            pose_similarity_list = calculate_landmarks_similarity(drive_idx, landmarks_list,image_shapes, landmark_start_idx, landmark_end_idx,top_k=top_k, ascending=True)
            src_idx = random.choice(pose_similarity_list)
            while abs(src_idx-drive_idx)<5:
                src_idx = random.choice(pose_similarity_list)
        except Exception as e:
            print(e)
            return None
    elif sample_method=="pose_similarity_and_closed_mouth":
        landmark_start_idx = 0
        landmark_end_idx = 16
        try:
            top_k = int(top_k_ratio*len(landmarks_list))
            closed_mouth_list = get_closed_mouth(landmarks_list, ascending=True,top_k=top_k)
            pose_similarity_list = calculate_landmarks_similarity(drive_idx, landmarks_list,image_shapes, landmark_start_idx, landmark_end_idx,top_k=top_k, ascending=True)
            common_list = list(set(closed_mouth_list).intersection(set(pose_similarity_list)))
            if len(common_list) == 0:
                src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)
            else:
                src_idx = random.choice(common_list)
            while abs(src_idx-drive_idx) <5:
                src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)
        except Exception as e:
            print(e)
            return None
    elif sample_method=="pose_similarity_and_mouth_dissimilarity":
        top_k = int(top_k_ratio*len(landmarks_list))
        try:
            top_k = int(top_k_ratio*len(landmarks_list))
            landmark_start_idx = 0
            landmark_end_idx = 16
            pose_similarity_list = calculate_landmarks_similarity(drive_idx, landmarks_list,image_shapes, landmark_start_idx, landmark_end_idx,top_k=top_k, ascending=True)
            landmark_start_idx = 60
            landmark_end_idx = 67
            mouth_dissimilarity_list = calculate_landmarks_similarity(drive_idx, landmarks_list,image_shapes, landmark_start_idx, landmark_end_idx,top_k=top_k, ascending=False)
            common_list = list(set(pose_similarity_list).intersection(set(mouth_dissimilarity_list)))
            if len(common_list) == 0:
                src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)
            else:
                src_idx = random.choice(common_list)
            while abs(src_idx-drive_idx) <5:
                src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)
        except Exception as e:
            print(e)
            return None
    else:
        raise ValueError(f"Unknown sample_method: {sample_method}")
    return src_idx

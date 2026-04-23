
import torch

import face_recognition

from typing import Dict, List
from utils import show_image


def _tensor_to_numpy(img: torch.Tensor):
    img = img.detach().cpu()
    if img.dim() == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    #values are uint8 in [0, 255]
    if img.dtype != torch.uint8:
        img = img.float()
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.clamp(0, 255).to(torch.uint8)

    return img.contiguous().numpy()


def detect_faces(img: torch.Tensor) -> List[List[float]]:
   
    detection_results: List[List[float]] = []

   

    img_np = _tensor_to_numpy(img)

    face_locations = face_recognition.face_locations(img_np, model='hog') 

    for (top, right, bottom, left) in face_locations:
        topleft_x = float(left)
        topleft_y = float(top)
        box_width = float(right - left)
        box_height = float(bottom - top)
        detection_results.append([topleft_x, topleft_y, box_width, box_height])


    return detection_results


def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    
    cluster_results: List[List[str]] = [[] for _ in range(K)]

    image_names: List[str] = []
    image_encodings: List[torch.Tensor] = []
    for img_name, img_tensor in imgs.items():
        image_names.append(img_name)

        img_np = _tensor_to_numpy(img_tensor)

        face_locations = face_recognition.face_locations(img_np, model='hog')

        if len(face_locations) == 0:
            image_encodings.append(None)
            continue

        encodings = face_recognition.face_encodings(
            img_np,
            known_face_locations=face_locations
        )

        if len(encodings) == 0:
            image_encodings.append(None)
        else:
            enc_tensor = torch.tensor(encodings[0], dtype=torch.float32)
            image_encodings.append(enc_tensor)

    valid_indices = [i for i, e in enumerate(image_encodings) if e is not None]
    invalid_indices = [i for i, e in enumerate(image_encodings) if e is None]

    if len(valid_indices) == 0:
        for name in image_names:
            cluster_results[0].append(name)
        return cluster_results

    valid_encodings = torch.stack([image_encodings[i] for i in valid_indices])
    M = valid_encodings.shape[0]
    actual_K = min(K, M)
    centroids = _kmeans_plus_plus_init(valid_encodings, actual_K)
    assignments = torch.full((M,), -1, dtype=torch.long)
    for _ in range(300):
        x_sq = (valid_encodings ** 2).sum(dim=1, keepdim=True)
        c_sq = (centroids ** 2).sum(dim=1, keepdim=True).T
        cross = valid_encodings @ centroids.T
        dists = (x_sq + c_sq - 2.0 * cross).clamp(min=0.0)
        new_assignments = dists.argmin(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for k in range(actual_K):
            mask = (assignments == k)
            if mask.any():
                centroids[k] = valid_encodings[mask].mean(dim=0)
    for idx_in_valid, img_idx in enumerate(valid_indices):
        cluster_id = int(assignments[idx_in_valid].item())
        cluster_results[cluster_id].append(image_names[img_idx])
    for img_idx in invalid_indices:
        cluster_results[0].append(image_names[img_idx])

    return cluster_results

def _kmeans_plus_plus_init(data: torch.Tensor, K: int) -> torch.Tensor:
    M, D = data.shape
    centroids = torch.zeros(K, D, dtype=data.dtype)
    first_idx = int(torch.randint(M, (1,)).item())
    centroids[0] = data[first_idx]

    for k in range(1, K):
        chosen = centroids[:k]
        x_sq = (data ** 2).sum(dim=1, keepdim=True)
        c_sq = (chosen ** 2).sum(dim=1, keepdim=True).T
        cross = data @ chosen.T
        dists = (x_sq + c_sq - 2.0 * cross).clamp(min=0.0)
        min_dists = dists.min(dim=1).values
        if float(min_dists.sum().item()) == 0.0:
            next_idx = int(torch.randint(M, (1,)).item())
        else:
            probs = min_dists / min_dists.sum()
            next_idx = int(torch.multinomial(probs, num_samples=1).item())
        centroids[k] = data[next_idx]
    return centroids
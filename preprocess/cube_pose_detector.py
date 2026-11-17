"""Offline AprilTag cube-pose detection for both hands."""
import itertools
import numpy as np
import cv2

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


APRILTAG_DICT    = cv2.aruco.DICT_APRILTAG_36H11
APRILTAG_DETECT_SCALES = (1.0, 2.0, 3.0)
CUBE_EDGE_M      = 0.046   # 46mm 正方体外边长 = 40mm tag + 两侧各 3mm 边
TAG_SIZE_M       = 0.040   # AprilTag 图案边长
REPROJ_THRESH_PX = 2.0
HALF_EDGE        = CUBE_EDGE_M / 2.0

# Pose selection guards.  Depth-only reconstruction/fallback is intentionally
# disabled; accepted tag poses come only from RGB AprilTag corner PnP.
DEPTH_MIN_M = 0.04
DEPTH_MAX_M = 2.50
DEPTH_PATCH_RADIUS_PX = 3
DEPTH_SCORE_SCALE_M = 0.030
CUBE_CENTER_INSIDE_MIN_M = HALF_EDGE * 0.50
CUBE_CENTER_PLANE_TOL_M = 0.006
CUBE_FACE_PLANE_TOL_M = 0.006
TAG_NORMAL_BACKFACE_REJECT_COS = 0.20
CUBE_CENTER_VIEW_DOT_MIN_M2 = -1e-8
CUBE_CENTER_VIEW_DOT_PENALTY_SCALE_M2 = 1e-4
CUBE_Z_MIN_M = 0.04
CUBE_Z_MAX_M = 2.50
CONTINUITY_POS_SCALE_M = 0.040
CONTINUITY_ROT_SCALE_RAD = 0.75
MULTITAG_POS_SPREAD_MAX_M = 0.025
MULTITAG_ROT_SPREAD_MAX_RAD = 0.45
MULTITAG_POS_SCORE_SCALE_M = 0.006
MULTITAG_ROT_SCORE_SCALE_RAD = 0.18

# 右手 cube: tag IDs 0-4；左手 cube: tag IDs 5-9.
# Face order is front(+Z), back(-Z), right(+X), left(-X), top(+Y).  The two
# hands are configured independently so a mirrored/rotated left cube can be
# fixed by changing LEFT_TAG_SPECS only.
RIGHT_FACE_TAG_IDS = [0, 1, 2, 3, 4]
LEFT_FACE_TAG_IDS  = [5, 6, 7, 8, 9]
RIGHT_TAG_ID_TO_FACE = {tid: fi for fi, tid in enumerate(RIGHT_FACE_TAG_IDS)}
LEFT_TAG_ID_TO_FACE  = {tid: fi for fi, tid in enumerate(LEFT_FACE_TAG_IDS)}
RIGHT_TAG_SPECS = {tid: (fi, 0) for tid, fi in RIGHT_TAG_ID_TO_FACE.items()}
LEFT_TAG_SPECS = {tid: (fi, 0) for tid, fi in LEFT_TAG_ID_TO_FACE.items()}

_HS = TAG_SIZE_M / 2.0
TAG_OBJ_PTS = np.array([
    [-_HS,  _HS, 0],
    [ _HS,  _HS, 0],
    [ _HS, -_HS, 0],
    [-_HS, -_HS, 0],
], dtype=np.float32)

# T_tag→cube（各面 tag 坐标系到正方体中心）
_h = HALF_EDGE
T_TAG_TO_CUBE = [None] * 5
T_TAG_TO_CUBE[0] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,_h],[0,0,0,1]], dtype=np.float64)
T_TAG_TO_CUBE[1] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,-_h],[0,0,0,1]], dtype=np.float64)
T_TAG_TO_CUBE[2] = np.array([[0,0,1,_h],[0,1,0,0],[-1,0,0,0],[0,0,0,1]], dtype=np.float64)
T_TAG_TO_CUBE[3] = np.array([[0,0,-1,-_h],[0,1,0,0],[1,0,0,0],[0,0,0,1]], dtype=np.float64)
T_TAG_TO_CUBE[4] = np.array([[1,0,0,0],[0,0,1,_h],[0,-1,0,0],[0,0,0,1]], dtype=np.float64)

T_CUBE_TO_TAG = []
for _T in T_TAG_TO_CUBE:
    _R, _t = _T[:3,:3], _T[:3,3]
    _Ti = np.eye(4, dtype=np.float64)
    _Ti[:3,:3] = _R.T
    _Ti[:3,3]  = -_R.T @ _t
    T_CUBE_TO_TAG.append(_Ti)


def _rz_quarter(k):
    a = (int(k) % 4) * np.pi / 2.0
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return T


def _cube_to_tag_from_spec(face_idx, rot_quarters=0):
    T_tag_to_cube = T_TAG_TO_CUBE[int(face_idx)] @ _rz_quarter(rot_quarters)
    return inv_T(T_tag_to_cube)


RIGHT_TAG_ID_TO_CUBE_TO_TAG = {
    tid: _cube_to_tag_from_spec(face_idx, rot_q)
    for tid, (face_idx, rot_q) in RIGHT_TAG_SPECS.items()
}
LEFT_TAG_ID_TO_CUBE_TO_TAG = {
    tid: _cube_to_tag_from_spec(face_idx, rot_q)
    for tid, (face_idx, rot_q) in LEFT_TAG_SPECS.items()
}

_CUBE_VERTEX_SIGNS = np.array([
    [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
    [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
], dtype=np.float64)
CUBE_VERTICES = _CUBE_VERTEX_SIGNS * HALF_EDGE


def _reproj_error(rvec, tvec, K, dist, img_pts):
    proj, _ = cv2.projectPoints(TAG_OBJ_PTS, rvec, tvec, K, dist)
    return float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1)))


def _matrix_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _rotation_angle(R_a, R_b):
    R = R_a.T @ R_b
    c = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _cube_vertices_to_cam(T_cube_to_cam):
    R = T_cube_to_cam[:3, :3]
    t = T_cube_to_cam[:3, 3]
    return (R @ CUBE_VERTICES.T).T + t


def _cube_center_view_dot(T_tag_to_cam, T_cube_to_cam):
    tag_center = T_tag_to_cam[:3, 3]
    cube_center = T_cube_to_cam[:3, 3]
    # A = cam->tag_center, B = cube_center.  A correct cube center is farther
    # along the camera ray than the visible tag center, so dot(B - A, A) >= 0.
    return float(np.dot(cube_center - tag_center, tag_center))


def _view_dot_penalty_against_entries(T_cube_to_cam, entries):
    penalty = 0.0
    worst_dot = np.inf
    for entry in entries:
        candidates = entry.get("candidates", [])
        if not candidates:
            continue
        # Use this visible tag's best candidate as its measured tag center.
        view_dot = _cube_center_view_dot(candidates[0]["T_tag_to_cam"], T_cube_to_cam)
        worst_dot = min(worst_dot, view_dot)
        if view_dot < CUBE_CENTER_VIEW_DOT_MIN_M2:
            penalty += (
                CUBE_CENTER_VIEW_DOT_MIN_M2 - view_dot
            ) / CUBE_CENTER_VIEW_DOT_PENALTY_SCALE_M2
    if np.isinf(worst_dot):
        worst_dot = 0.0
    return penalty, float(worst_dot)


def _tag_plane_inside_metrics(T_tag_to_cam, T_cube_to_cam):
    tag_center = T_tag_to_cam[:3, 3]
    tag_normal = T_tag_to_cam[:3, 2]
    normal_norm = float(np.linalg.norm(tag_normal))
    if normal_norm < 1e-9:
        return None
    tag_normal = tag_normal / normal_norm
    cube_center = T_cube_to_cam[:3, 3]
    center_signed = float(np.dot(cube_center - tag_center, tag_normal))
    verts = _cube_vertices_to_cam(T_cube_to_cam)
    vertex_signed = (verts - tag_center) @ tag_normal
    return center_signed, float(np.max(vertex_signed)), float(np.min(vertex_signed))


def _cube_pose_is_inside_tag_plane(T_tag_to_cam, T_cube_to_cam, strict_center=True):
    metrics = _tag_plane_inside_metrics(T_tag_to_cam, T_cube_to_cam)
    if metrics is None:
        return False
    center_signed, vertex_max, _ = metrics
    if center_signed > -CUBE_CENTER_INSIDE_MIN_M:
        return False
    if strict_center and abs(center_signed + HALF_EDGE) > CUBE_CENTER_PLANE_TOL_M:
        return False
    if vertex_max > CUBE_FACE_PLANE_TOL_M:
        return False
    return True


def _median_depth_at(depth, u, v, radius=DEPTH_PATCH_RADIUS_PX):
    if depth is None or getattr(depth, "ndim", 0) < 2:
        return None
    h, w = depth.shape[:2]
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    if ui < 0 or ui >= w or vi < 0 or vi >= h:
        return None
    x0 = max(0, ui - radius)
    x1 = min(w, ui + radius + 1)
    y0 = max(0, vi - radius)
    y1 = min(h, vi + radius + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
    valid = patch[
        np.isfinite(patch)
        & (patch > DEPTH_MIN_M)
        & (patch < DEPTH_MAX_M)
    ]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _marker_depth_hint(img_pts, depth):
    samples = []
    center = img_pts.mean(axis=0)
    for u, v in [center, *img_pts]:
        d = _median_depth_at(depth, u, v)
        if d is not None:
            samples.append(d)
    if not samples:
        return None
    return float(np.median(samples))


def _score_tag_candidate(T_tag_to_cam, T_cube_to_tag, reproj_err, img_pts, depth, prev_cube_T):
    T_cube_to_cam = T_tag_to_cam @ T_cube_to_tag
    tag_z = float(T_tag_to_cam[2, 3])
    cube_z = float(T_cube_to_cam[2, 3])
    if not np.isfinite(tag_z) or not np.isfinite(cube_z):
        return None
    if cube_z < CUBE_Z_MIN_M or cube_z > CUBE_Z_MAX_M:
        return None
    tag_center_cam = T_tag_to_cam[:3, 3]
    center_norm = float(np.linalg.norm(tag_center_cam))
    if center_norm < 1e-9:
        return None

    # Main disambiguation: A = tag center in camera coordinates, B = cube center
    # in camera coordinates.  A correct cube center lies behind the visible tag
    # along the camera ray, so dot(B - A, A) is non-negative.
    view_dot = _cube_center_view_dot(T_tag_to_cam, T_cube_to_cam)
    if view_dot < CUBE_CENTER_VIEW_DOT_MIN_M2:
        return None

    score = float(reproj_err)

    if prev_cube_T is not None:
        pos_jump = float(np.linalg.norm(T_cube_to_cam[:3, 3] - prev_cube_T[:3, 3]))
        rot_jump = _rotation_angle(prev_cube_T[:3, :3], T_cube_to_cam[:3, :3])
        score += min(pos_jump / CONTINUITY_POS_SCALE_M, 8.0)
        score += min(rot_jump / CONTINUITY_ROT_SCALE_RAD, 4.0)

    return {
        "score": score,
        "T_tag_to_cam": T_tag_to_cam,
        "T_cube_to_cam": T_cube_to_cam,
        "reproj_err": float(reproj_err),
        "view_dot": view_dot,
    }


def _tag_pose_candidates(img_pts, K, dist, depth, T_cube_to_tag, prev_cube_T):
    candidates = []
    try:
        ret = cv2.solvePnPGeneric(
            TAG_OBJ_PTS, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        ok = bool(ret[0])
        rvecs = ret[1] if ok and len(ret) > 1 else []
        tvecs = ret[2] if ok and len(ret) > 2 else []
    except Exception:
        rvecs, tvecs = [], []

    for rvec, tvec in zip(rvecs, tvecs):
        err = _reproj_error(rvec, tvec, K, dist, img_pts)
        if err > REPROJ_THRESH_PX:
            continue
        T_tag_to_cam = _matrix_from_rvec_tvec(rvec, tvec)
        cand = _score_tag_candidate(
            T_tag_to_cam, T_cube_to_tag, err, img_pts, depth, prev_cube_T
        )
        if cand is not None:
            candidates.append(cand)

    return sorted(candidates, key=lambda c: c["score"])


def _select_tag_pose_candidate(img_pts, K, dist, depth, T_cube_to_tag, prev_cube_T):
    candidates = _tag_pose_candidates(img_pts, K, dist, depth, T_cube_to_tag, prev_cube_T)
    return candidates[0] if candidates else None


def fuse_cube_pose(face_indices, T_tags_to_cam, weights=None):
    n = len(face_indices)
    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.array(weights); w /= w.sum()
    t_avg = np.zeros(3); R_mean = np.zeros((3,3))
    for wi, fi, T_tc in zip(w, face_indices, T_tags_to_cam):
        T_cc = T_tc @ T_CUBE_TO_TAG[fi]
        t_avg += wi * T_cc[:3,3]
        R_mean += wi * T_cc[:3,:3]
    U, _, Vt = np.linalg.svd(R_mean)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:,-1] *= -1; R_avg = U @ Vt
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R_avg; T[:3,3] = t_avg
    return T


def _fuse_cube_pose_matrices(T_cubes_to_cam, weights=None):
    n = len(T_cubes_to_cam)
    if n == 0:
        return None
    if weights is None:
        w = np.ones(n, dtype=np.float64) / float(n)
    else:
        w = np.asarray(weights, dtype=np.float64)
        s = float(np.sum(w))
        if s <= 1e-12:
            w = np.ones(n, dtype=np.float64) / float(n)
        else:
            w /= s
    t_avg = np.zeros(3, dtype=np.float64)
    R_mean = np.zeros((3, 3), dtype=np.float64)
    for wi, T_cc in zip(w, T_cubes_to_cam):
        t_avg += wi * T_cc[:3, 3]
        R_mean += wi * T_cc[:3, :3]
    U, _, Vt = np.linalg.svd(R_mean)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_avg
    T[:3, 3] = t_avg
    return T


def _combo_consistency(combo, fused_T):
    pos_res = []
    rot_res = []
    view_dot_penalty = 0.0
    for cand in combo:
        T_cc = cand["T_cube_to_cam"]
        pos_res.append(float(np.linalg.norm(T_cc[:3, 3] - fused_T[:3, 3])))
        rot_res.append(_rotation_angle(T_cc[:3, :3], fused_T[:3, :3]))
        view_dot = _cube_center_view_dot(cand["T_tag_to_cam"], fused_T)
        if view_dot < CUBE_CENTER_VIEW_DOT_MIN_M2:
            view_dot_penalty += (
                CUBE_CENTER_VIEW_DOT_MIN_M2 - view_dot
            ) / CUBE_CENTER_VIEW_DOT_PENALTY_SCALE_M2
    pos_max = max(pos_res) if pos_res else 0.0
    rot_max = max(rot_res) if rot_res else 0.0
    return pos_max, rot_max, view_dot_penalty


def _best_multitag_combo(entries):
    global_best = None
    n = len(entries)
    # Prefer the largest internally consistent visible subset.  This lets a bad
    # side tag lose to two mutually consistent tags without discarding all data.
    for subset_size in range(n, 1, -1):
        subset_best = None
        for subset in itertools.combinations(entries, subset_size):
            cand_lists = [entry["candidates"][:2] for entry in subset]
            for combo_tuple in itertools.product(*cand_lists):
                combo = list(combo_tuple)
                weights = [1.0 / max(c["reproj_err"], 0.05) for c in combo]
                fused_T = _fuse_cube_pose_matrices(
                    [c["T_cube_to_cam"] for c in combo], weights
                )
                pos_max, rot_max, inside_penalty = _combo_consistency(combo, fused_T)
                view_dot_penalty, worst_view_dot = _view_dot_penalty_against_entries(
                    fused_T, entries
                )
                inside_penalty += view_dot_penalty
                score = (
                    inside_penalty
                    + pos_max / MULTITAG_POS_SCORE_SCALE_M
                    + rot_max / MULTITAG_ROT_SCORE_SCALE_RAD
                    + float(np.mean([c["score"] for c in combo]))
                    - 0.25 * subset_size
                )
                record = {
                    "score": score,
                    "T_cube_to_cam": fused_T,
                    "subset_size": subset_size,
                    "pos_max": pos_max,
                    "rot_max": rot_max,
                    "inside_penalty": inside_penalty,
                    "combo": combo,
                    "worst_view_dot": worst_view_dot,
                }
                if subset_best is None or record["score"] < subset_best["score"]:
                    subset_best = record
                if global_best is None or record["score"] < global_best["score"]:
                    global_best = record
        if (
            subset_best is not None
            and subset_best["inside_penalty"] <= 1e-6
            and subset_best["pos_max"] <= MULTITAG_POS_SPREAD_MAX_M
            and subset_best["rot_max"] <= MULTITAG_ROT_SPREAD_MAX_RAD
        ):
            return subset_best
    return global_best


def _select_hand_cube_pose(entries):
    if not entries:
        return None
    if len(entries) >= 2:
        best = _best_multitag_combo(entries)
        if (
            best is not None
            and best["inside_penalty"] <= 1e-6
            and best["pos_max"] <= MULTITAG_POS_SPREAD_MAX_M
            and best["rot_max"] <= MULTITAG_ROT_SPREAD_MAX_RAD
        ):
            return best["T_cube_to_cam"]

    # Single-tag fallback is intentional: a single good AprilTag frame is still
    # useful, but it must already have passed the tag-plane inside constraint.
    singles = [cand for entry in entries for cand in entry["candidates"]]
    if not singles:
        return None
    for cand in sorted(singles, key=lambda c: c["score"]):
        penalty, _ = _view_dot_penalty_against_entries(cand["T_cube_to_cam"], entries)
        if penalty <= 1e-6:
            return cand["T_cube_to_cam"]
    return None


def make_apriltag_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, params)


def _detect_markers_multiscale(gray, detector):
    by_id = {}
    for scale in APRILTAG_DETECT_SCALES:
        if scale == 1.0:
            detect_gray = gray
        else:
            detect_gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
        corners, marker_ids, _ = detector.detectMarkers(detect_gray)
        if marker_ids is None or len(marker_ids) == 0:
            continue
        for corner, marker_id in zip(corners, marker_ids.flatten()):
            tag_id = int(marker_id)
            pts = corner.reshape(4, 2).astype(np.float32)
            if scale != 1.0:
                pts /= float(scale)
            # Same physical tag can be found at several scales.  Prefer the
            # first hit, which comes from the lowest-cost / least-resampled image.
            if tag_id not in by_id:
                by_id[tag_id] = pts.reshape(1, 4, 2)
        has_right = any(tid in RIGHT_TAG_ID_TO_FACE for tid in by_id)
        has_left = any(tid in LEFT_TAG_ID_TO_FACE for tid in by_id)
        if has_right and has_left:
            break
    if not by_id:
        return [], None
    ids = np.array([[tid] for tid in by_id.keys()], dtype=np.int32)
    return list(by_id.values()), ids


def detect_both_cube_poses(rgb, detector, K, dist, depth=None,
                           prev_right_T=None, prev_left_T=None):
    """
    检测双手 AprilTag，分别融合右手(IDs 0-4)和左手(IDs 5-9)的正方体位姿。
    返回 (T_right, T_left)，未检测到则为 None。
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, marker_ids = _detect_markers_multiscale(gray, detector)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None

    right_entries = []
    left_entries = []

    for i, tag_id in enumerate(marker_ids.flatten()):
        tag_id = int(tag_id)
        # 判断属于哪只手
        if tag_id in RIGHT_TAG_ID_TO_FACE:
            hand_entries = right_entries
            fi = RIGHT_TAG_ID_TO_FACE[tag_id]
            T_cube_to_tag = RIGHT_TAG_ID_TO_CUBE_TO_TAG[tag_id]
            prev_cube_T = prev_right_T
        elif tag_id in LEFT_TAG_ID_TO_FACE:
            hand_entries = left_entries
            fi = LEFT_TAG_ID_TO_FACE[tag_id]
            T_cube_to_tag = LEFT_TAG_ID_TO_CUBE_TO_TAG[tag_id]
            prev_cube_T = prev_left_T
        else:
            continue

        img_pts = corners[i].reshape(4, 2).astype(np.float32)
        candidates = _tag_pose_candidates(
            img_pts, K, dist, depth, T_cube_to_tag, prev_cube_T
        )
        if not candidates:
            continue
        hand_entries.append({
            "tag_id": tag_id,
            "face_idx": fi,
            "candidates": candidates,
        })

    T_right = _select_hand_cube_pose(right_entries)
    T_left = _select_hand_cube_pose(left_entries)
    return T_right, T_left

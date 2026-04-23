import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W, H = 1280, 720
CX, CY = W // 2, H // 2

# ── State ──
angle_x, angle_y, angle_z = 30.0, 30.0, 0.0
scale = 220.0
prev_ix, prev_iy = None, None
prev_dist = None
auto_rotate = False
auto_speed = 0.5
wireframe_only = False
show_axes = True
show_grid = True
glow_effect = True
mode = "ROTATE"
selected_face = -1
face_colors_custom = {}
draw_strokes = []
drawing_3d = False
prev_draw_pt = None
last_gesture_time = 0
gesture_cooldown = 0.6
menu_active = False
brightness = 1.0
explode = 0.0
explode_dir = 1
particle_list = []
screenshot_flash = 0

VERTICES = np.array([
    [-1, -1, -1], [ 1, -1, -1],
    [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1],
    [ 1,  1,  1], [-1,  1,  1],
], dtype=float)

EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

FACES = [
    (0,1,2,3), (4,5,6,7),
    (0,1,5,4), (2,3,7,6),
    (0,3,7,4), (1,2,6,5),
]

FACE_NAMES = ["DEPAN","BELAKANG","BAWAH","ATAS","KIRI","KANAN"]

DEFAULT_FACE_COLORS = [
    (0, 180, 255), (0, 140, 220),
    (0, 100, 200), (0, 80, 180),
    (0, 120, 210), (0, 60, 160),
]

COLOR_PALETTE = [
    (0, 180, 255), (0, 255, 100),
    (255, 60, 60),  (255, 200, 0),
    (200, 0, 255),  (0, 255, 220),
    (255, 120, 0),  (255, 255, 255),
]

MODES = ["ROTATE", "DRAW", "COLOR", "EXPLODE", "WIRE", "AXES"]
mode_idx = 0

def rotate_x(pts, a):
    c, s = math.cos(a), math.sin(a)
    R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return pts @ R.T

def rotate_y(pts, a):
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return pts @ R.T

def rotate_z(pts, a):
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return pts @ R.T

def project_pts(pts, sc, ox=0, oy=0):
    result = []
    for p in pts:
        z = p[2] + 4
        if z < 0.1: z = 0.1
        x2d = int(CX + ox + (p[0] * sc) / z)
        y2d = int(CY + oy + (p[1] * sc) / z)
        result.append((x2d, y2d))
    return result

def get_transformed(verts, ex=0.0):
    pts = verts.copy()
    if ex > 0:
        offsets = []
        for face in FACES:
            center = np.mean([pts[v] for v in face], axis=0)
            offsets.append(center * ex * 0.5)
        pts_exp = []
        for vi in range(len(pts)):
            face_contributions = []
            for fi, face in enumerate(FACES):
                if vi in face:
                    face_contributions.append(offsets[fi])
            if face_contributions:
                pts_exp.append(pts[vi] + np.mean(face_contributions, axis=0))
            else:
                pts_exp.append(pts[vi])
        pts = np.array(pts_exp)
    pts = rotate_x(pts, math.radians(angle_x))
    pts = rotate_y(pts, math.radians(angle_y))
    pts = rotate_z(pts, math.radians(angle_z))
    return pts

def jari_terangkat(lm, ujung, pangkal):
    return lm[ujung].y < lm[pangkal].y

def hitung_jarak(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_face_normal(pts_3d, face):
    v0 = pts_3d[face[0]]
    v1 = pts_3d[face[1]]
    v2 = pts_3d[face[2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    return n / (np.linalg.norm(n) + 1e-9)

def apply_brightness(color, br):
    return tuple(min(255, int(c * br)) for c in color)

def add_particles(cx, cy, color, n=8):
    for _ in range(n):
        angle = np.random.uniform(0, 2*math.pi)
        speed = np.random.uniform(2, 8)
        life = np.random.uniform(20, 40)
        particle_list.append({
            "x": cx, "y": cy,
            "vx": math.cos(angle)*speed,
            "vy": math.sin(angle)*speed,
            "life": life, "max_life": life,
            "color": color, "size": np.random.randint(2, 6)
        })

def update_particles(frame):
    dead = []
    for p in particle_list:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.2
        p["life"] -= 1
        alpha = p["life"] / p["max_life"]
        color = tuple(int(c * alpha) for c in p["color"])
        size = max(1, int(p["size"] * alpha))
        cv2.circle(frame, (int(p["x"]), int(p["y"])), size, color, -1)
        if p["life"] <= 0:
            dead.append(p)
    for d in dead:
        particle_list.remove(d)

def gambar_background(frame):
    frame[:] = (8, 8, 12)
    if show_grid:
        for i in range(0, W, 80):
            cv2.line(frame, (i, 0), (i, H), (0, 25, 35), 1)
        for i in range(0, H, 80):
            cv2.line(frame, (0, i), (W, i), (0, 25, 35), 1)
    cv2.circle(frame, (CX, CY), 3, (0, 80, 120), -1)

def gambar_axes(frame, projected, pts_3d, sc):
    origin = (CX, CY)
    axis_pts = np.array([[1.5,0,0],[0,1.5,0],[0,0,1.5]], dtype=float)
    axis_pts = rotate_x(axis_pts, math.radians(angle_x))
    axis_pts = rotate_y(axis_pts, math.radians(angle_y))
    axis_pts = rotate_z(axis_pts, math.radians(angle_z))
    axis_proj = project_pts(axis_pts, sc)
    labels = ["X","Y","Z"]
    colors = [(0,80,255),(0,200,80),(255,80,0)]
    for i, (ap, col, lbl) in enumerate(zip(axis_proj, colors, labels)):
        cv2.arrowedLine(frame, origin, ap, col, 2, tipLength=0.3)
        cv2.putText(frame, lbl, (ap[0]+5, ap[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

def gambar_cube(frame):
    global selected_face
    pts_3d = get_transformed(VERTICES, explode)
    projected = project_pts(pts_3d, scale)

    face_data = []
    for fi, face in enumerate(FACES):
        z_avg = np.mean([pts_3d[v][2] for v in face])
        normal = get_face_normal(pts_3d, face)
        light_dir = np.array([0.5, -0.8, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        diffuse = max(0.3, np.dot(normal, light_dir))
        face_data.append((z_avg, fi, face, diffuse))

    face_data.sort(key=lambda x: x[0])

    for z_avg, fi, face, diffuse in face_data:
        pts_face = np.array([projected[v] for v in face], dtype=np.int32)
        base_color = face_colors_custom.get(fi, DEFAULT_FACE_COLORS[fi])
        lit_color = apply_brightness(base_color, diffuse * brightness)

        if not wireframe_only:
            overlay = frame.copy()
            alpha_face = 0.18 + diffuse * 0.12
            if fi == selected_face:
                alpha_face = 0.4
                lit_color = apply_brightness(base_color, 1.5 * brightness)
            cv2.fillPoly(overlay, [pts_face], lit_color)
            cv2.addWeighted(overlay, alpha_face, frame, 1 - alpha_face, 0, frame)

        edge_bright = apply_brightness(base_color, diffuse * brightness * 1.3)
        edge_w = 3 if fi == selected_face else 2
        cv2.polylines(frame, [pts_face], True, edge_bright, edge_w, cv2.LINE_AA)

        if glow_effect and not wireframe_only:
            glow_color = apply_brightness(base_color, diffuse * 0.4 * brightness)
            cv2.polylines(frame, [pts_face], True, glow_color, 6, cv2.LINE_AA)

        if fi == selected_face:
            cx_f = int(np.mean([projected[v][0] for v in face]))
            cy_f = int(np.mean([projected[v][1] for v in face]))
            cv2.putText(frame, FACE_NAMES[fi], (cx_f - 25, cy_f),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for edge in EDGES:
        p1, p2 = projected[edge[0]], projected[edge[1]]
        z_avg = (pts_3d[edge[0]][2] + pts_3d[edge[1]][2]) / 2
        br = int(120 + (z_avg + 1.5) * 40)
        br = max(60, min(255, int(br * brightness)))
        cv2.line(frame, p1, p2, (0, br, 255), 2, cv2.LINE_AA)

    for i, p in enumerate(projected):
        z = pts_3d[i][2]
        bright = int(150 + (z + 1.5) * 40)
        bright = max(80, min(255, int(bright * brightness)))
        cv2.circle(frame, p, 5, (0, bright, 255), -1, cv2.LINE_AA)
        if glow_effect:
            cv2.circle(frame, p, 9, (0, bright//2, 128), 1, cv2.LINE_AA)

    if show_axes:
        gambar_axes(frame, projected, pts_3d, scale)

    for stroke in draw_strokes:
        for i in range(1, len(stroke["pts"])):
            cv2.line(frame, stroke["pts"][i-1], stroke["pts"][i],
                     stroke["color"], stroke["width"], cv2.LINE_AA)

def gambar_hud(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 90), (5, 8, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "HOLOGRAPHIC  INTERFACE  v2.0",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    info = f"MODE:{MODES[mode_idx]}  ROT:{angle_x:.0f}/{angle_y:.0f}/{angle_z:.0f}  SCALE:{scale:.0f}  AUTO:{'ON' if auto_rotate else 'OFF'}  GLOW:{'ON' if glow_effect else 'OFF'}"
    cv2.putText(frame, info, (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 150, 200), 1)

    cv2.putText(frame, "TELUNJUK=AKSI  2TANGAN=ZOOM  MENGEPAL=RESET  PINCH=GANTI MODE",
                (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 160), 1)

    bar_w = 300
    cv2.rectangle(frame, (W-bar_w-20, 10), (W-20, 80), (10, 15, 25), -1)
    cv2.rectangle(frame, (W-bar_w-20, 10), (W-20, 80), (0, 60, 100), 1)
    cv2.putText(frame, "MODES:", (W-bar_w-10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 180), 1)

    mw = bar_w // len(MODES)
    for i, m in enumerate(MODES):
        x = W - bar_w - 20 + i * mw
        color = (0, 200, 255) if i == mode_idx else (0, 60, 100)
        cv2.rectangle(frame, (x+2, 38), (x+mw-2, 72), color, -1 if i == mode_idx else 1)
        cv2.putText(frame, m[:4], (x+4, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0) if i == mode_idx else (0, 120, 180), 1)

    if mode_idx == 2:
        pw = 36
        for i, col in enumerate(COLOR_PALETTE):
            x = 20 + i * (pw + 4)
            cv2.rectangle(frame, (x, H-65), (x+pw, H-35), col, -1)
            cv2.rectangle(frame, (x, H-65), (x+pw, H-35), (255,255,255), 1)

def gambar_scanline(frame):
    t = time.time()
    y = int((t * 120) % H)
    cv2.line(frame, (0, y), (W, y), (0, 50, 80), 1)
    if screenshot_flash > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (W,H), (255,255,255), -1)
        cv2.addWeighted(overlay, screenshot_flash/10, frame, 1-screenshot_flash/10, 0, frame)


def process_gesture(tangan_list, frame):
    global angle_x, angle_y, angle_z, scale
    global prev_ix, prev_iy, prev_dist
    global mode_idx, auto_rotate, glow_effect
    global wireframe_only, show_axes, show_grid
    global selected_face, drawing_3d, prev_draw_pt
    global last_gesture_time, brightness, explode, explode_dir
    global screenshot_flash

    now = time.time()

    if len(tangan_list) == 1:
        t = tangan_list[0]
        lm = t["lm"]
        ix, iy = t["ix"], t["iy"]
        tx, ty = t["tx"], t["ty"]

        telunjuk_up  = jari_terangkat(lm, 8, 6)
        tengah_up    = jari_terangkat(lm, 12, 10)
        manis_up     = jari_terangkat(lm, 16, 14)
        kelingking_up= jari_terangkat(lm, 20, 18)
        jempol_up    = lm[4].x < lm[3].x

        mengepal = not telunjuk_up and not tengah_up and not manis_up and not kelingking_up
        pinch_dist = hitung_jarak((ix, iy), (tx, ty))
        pinching = pinch_dist < 40

        if mengepal:
            angle_x, angle_y, angle_z = 30.0, 30.0, 0.0
            scale = 220.0
            explode = 0.0
            prev_ix, prev_iy = None, None
            add_particles(CX, CY, (0, 200, 255), 20)
            cv2.putText(frame, "RESET", (CX-50, CY),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 200), 3)
            return

        if pinching and now - last_gesture_time > gesture_cooldown:
            mode_idx = (mode_idx + 1) % len(MODES)
            last_gesture_time = now
            add_particles(ix, iy, COLOR_PALETTE[mode_idx % len(COLOR_PALETTE)], 12)

        current_mode = MODES[mode_idx]

        if current_mode == "ROTATE" and telunjuk_up and not tengah_up:
            if prev_ix is not None:
                dx = ix - prev_ix
                dy = iy - prev_iy
                angle_y += dx * 0.4
                angle_x += dy * 0.4
            prev_ix, prev_iy = ix, iy
            cv2.line(frame, (CX, CY), (ix, iy), (0, 80, 160), 1)
            cv2.putText(frame, "ROTATING", (ix+15, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        elif current_mode == "ROTATE" and telunjuk_up and tengah_up and not manis_up:
            angle_z += (ix - (prev_ix or ix)) * 0.3
            prev_ix, prev_iy = ix, iy
            cv2.putText(frame, "ROLL", (ix+15, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        elif current_mode == "DRAW" and telunjuk_up and not tengah_up:
            cv2.circle(frame, (ix, iy), 8, (0, 255, 150), -1)
            if prev_draw_pt is not None:
                if not draw_strokes or len(draw_strokes[-1]["pts"]) == 0:
                    draw_strokes.append({"pts": [prev_draw_pt], "color": (0,255,150), "width": 3})
                draw_strokes[-1]["pts"].append((ix, iy))
            prev_draw_pt = (ix, iy)
            cv2.putText(frame, "DRAWING", (ix+15, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 1)

        elif current_mode == "DRAW" and telunjuk_up and tengah_up:
            prev_draw_pt = None
            if draw_strokes:
                draw_strokes.append({"pts": [], "color": (0,255,150), "width": 3})

        elif current_mode == "COLOR" and telunjuk_up:
            for fi, face in enumerate(FACES):
                pts_3d = get_transformed(VERTICES, explode)
                projected = project_pts(pts_3d, scale)
                pts_face = np.array([projected[v] for v in face], dtype=np.int32)
                if cv2.pointPolygonTest(pts_face, (ix, iy), False) >= 0:
                    selected_face = fi
                    if iy > H - 70:
                        color_idx = (ix - 20) // 40
                        if 0 <= color_idx < len(COLOR_PALETTE):
                            face_colors_custom[fi] = COLOR_PALETTE[color_idx]
                            add_particles(ix, iy, COLOR_PALETTE[color_idx], 15)

        elif current_mode == "EXPLODE" and telunjuk_up:
            if prev_iy is not None:
                dy = iy - prev_iy
                explode = max(0.0, min(2.0, explode - dy * 0.01))
            prev_ix, prev_iy = ix, iy
            cv2.putText(frame, f"EXPLODE:{explode:.2f}", (ix+15, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 1)

        elif current_mode == "WIRE" and telunjuk_up and now - last_gesture_time > gesture_cooldown:
            wireframe_only = not wireframe_only
            glow_effect = not wireframe_only
            last_gesture_time = now

        elif current_mode == "AXES" and telunjuk_up and now - last_gesture_time > gesture_cooldown:
            show_axes = not show_axes
            show_grid = not show_grid
            last_gesture_time = now

        if current_mode not in ["DRAW", "COLOR", "EXPLODE"]:
            prev_draw_pt = None
            if current_mode not in ["ROTATE"]:
                prev_ix, prev_iy = None, None

        if jempol_up and not telunjuk_up and now - last_gesture_time > gesture_cooldown:
            auto_rotate = not auto_rotate
            last_gesture_time = now
            cv2.putText(frame, f"AUTO {'ON' if auto_rotate else 'OFF'}",
                        (CX-60, CY-60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,200), 2)

    elif len(tangan_list) == 2:
        p1 = (tangan_list[0]["ix"], tangan_list[0]["iy"])
        p2 = (tangan_list[1]["ix"], tangan_list[1]["iy"])
        dist = hitung_jarak(p1, p2)
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)

        cv2.line(frame, p1, p2, (0, 200, 255), 2)
        cv2.circle(frame, mid, 8, (0, 255, 200), -1)

        if prev_dist is not None:
            delta = dist - prev_dist
            scale = max(80, min(600, scale + delta * 0.8))
            lm0 = tangan_list[0]["lm"]
            lm1 = tangan_list[1]["lm"]
            dy0 = lm0[8].y - lm0[6].y
            dy1 = lm1[8].y - lm1[6].y
            if dy0 < 0 and dy1 < 0:
                brightness = max(0.3, min(2.0, brightness + 0.01))
            elif dy0 > 0 and dy1 > 0:
                brightness = max(0.3, min(2.0, brightness - 0.01))

        prev_dist = dist
        prev_ix, prev_iy = None, None
        cv2.putText(frame, f"ZOOM:{scale:.0f}  BRIGHT:{brightness:.1f}",
                    (mid[0]+10, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    else:
        prev_dist = None


print("=" * 50)
print("HOLOGRAPHIC INTERFACE v2.0")
print("=" * 50)
print("GESTURE CONTROL:")
print("  ☝  Telunjuk       = Aksi sesuai mode")
print("  ✌  Telunjuk+Tengah = Aksi tambahan")
print("  👌 Pinch           = Ganti mode")
print("  👍 Jempol          = Toggle auto-rotate")
print("  ✊ Mengepal        = Reset semua")
print("  🤲 2 tangan        = Zoom & brightness")
print("MODES: ROTATE | DRAW | COLOR | EXPLODE | WIRE | AXES")
print("Tekan Q untuk keluar")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = hands.process(rgb)

    gambar_background(frame)

    if auto_rotate:
        angle_y += auto_speed
        angle_x += auto_speed * 0.3

    if screenshot_flash > 0:
        screenshot_flash -= 1

    gambar_cube(frame)
    update_particles(frame)
    gambar_scanline(frame)

    tangan_list = []
    if hasil.multi_hand_landmarks:
        for hand_landmarks in hasil.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 150, 255), thickness=2)
            )
            lm = hand_landmarks.landmark
            ix = int(lm[8].x * W)
            iy = int(lm[8].y * H)
            tx = int(lm[4].x * W)
            ty = int(lm[4].y * H)
            tangan_list.append({"ix": ix, "iy": iy, "tx": tx, "ty": ty, "lm": lm})
            cv2.circle(frame, (ix, iy), 12, (0, 255, 200), -1)
            cv2.circle(frame, (ix, iy), 18, (0, 150, 255), 2)

        process_gesture(tangan_list, frame)

    gambar_hud(frame)

    cv2.imshow("Holographic Interface v2.0", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"jarvis_{int(time.time())}.png"
        cv2.imwrite(fname, frame)
        screenshot_flash = 10
        print(f"Screenshot: {fname}")
    elif key == ord('c'):
        draw_strokes.clear()
        face_colors_custom.clear()
    elif key == ord('r'):
        angle_x, angle_y, angle_z = 30.0, 30.0, 0.0
        scale = 220.0
        explode = 0.0

cap.release()
cv2.destroyAllWindows()
